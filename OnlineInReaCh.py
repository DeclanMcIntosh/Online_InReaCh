import os 
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

from typing import List
import faiss 
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from skimage.measure import block_reduce
import time
import git
import csv  
import tqdm
import json
import argparse

from FeatureDescriptors import *
from utils import *
from model import *
from mvtec_loader import *


class OnlineInReaCh():
    def __init__(self, 
                 sample_image: np.ndarray, 
                 model : torch.nn.Module,
                 min_channel_length: int = 2,
                 filter_size: float = 13,
                 time_to_live: int = 3,
                 pos_weight: float = 0.0,
                 update_interval: int = 1,
                 quite: bool = False,
                 mode: str = 'testing',
                 **kwargs) -> None:
        
        self.quite = quite
        self.image_size = tuple(sample_image.shape)
        self.model = model
        self.pos_weight = pos_weight
        self.filter_size = filter_size
        self.min_channel_length = min_channel_length
        self.time_to_live = time_to_live
        self.update_interval = update_interval
        self.mode = mode

        # Setup feature Extraction 
        self.fd_gen = Feautre_Descriptor(model=model, image_size=self.image_size, **kwargs)
        self.scale = 4 # TODO not hard code this
        # Setup Channels 
        self.channels = []
        self.channel_seeds = []
        self.count = 0 

        # Logging
        self.scores = [] 
        self.t_masks = [] 
        self.inferance_speed = [] 
        self.update_speed = [] 
        self.image_wise_predictions = [] 
        self.image_wise_actual = [] 
        self.pixel_wise_AUROC = [] 
        self.channel_percision = []
        self.num_channels = [] # TODO
        self.feature_extraction_time = [] 

        # Faiss
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        self.nn_object = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 1024, flat_config)

    def gen_assoc(self, targets: torch.Tensor, 
                         sources: torch.Tensor, 
                         target_img_index: int, 
                         source_img_indexs: int):
        t_len = targets.size()[1]
        s_len = sources.size()[1]
        sources_zero_axis_min   = torch.from_numpy(np.ones(shape=(t_len))*np.inf).cuda()
        sources_zero_axis_index = torch.from_numpy(np.zeros(shape=(t_len))).cuda()
        targets_ones_axis_min   = torch.from_numpy(np.ones(shape=(s_len))*np.inf).cuda()
        targets_ones_axis_index = torch.from_numpy(np.zeros(shape=(s_len))).cuda()

        # Handle not having enough GPU memory to do everything in one big batch
        # This brute forces the distances measures no fancy faiss 
        # We are looking for the index to the minimum association from the new patches to the old channels and vice versa
        aval_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        max_side = int(np.floor(np.sqrt(aval_mem//32)))
        for x in range(int(np.ceil(s_len/max_side))):
            for y in range(int(np.ceil(t_len/max_side))):

                # Measure the distances in a small rectangle of the overall distance matrix  
                distances = measure_distances(sources[:,x*max_side:min([(x+1)*max_side,s_len])],
                    targets[:,y*max_side:min([(y+1)*max_side,t_len])])

                # Find the minimum association index to the sources 
                mins, args = (torch.min(distances,axis=0))
                sources_zero_axis_index[y*max_side:min([(y+1)*max_side,t_len])] = torch.where(
                    sources_zero_axis_min[y*max_side:min([(y+1)*max_side,t_len])] >= mins,
                    args + x*max_side,
                    sources_zero_axis_index[y*max_side:min([(y+1)*max_side,t_len])] 
                )
                # Find the minimum association distance to the sources
                sources_zero_axis_min[y*max_side:min([(y+1)*max_side,t_len])] = torch.minimum(
                    sources_zero_axis_min[y*max_side:min([(y+1)*max_side,t_len])],
                    mins
                )

                # Find the minimum association index to the targets 
                mins, args = (torch.min(distances,axis=1))
                targets_ones_axis_index[x*max_side:min([(x+1)*max_side,s_len])] = torch.where(
                    targets_ones_axis_min[x*max_side:min([(x+1)*max_side,s_len])] >= mins,
                    args + y*max_side,
                    targets_ones_axis_index[x*max_side:min([(x+1)*max_side,s_len])]
                )
                # Find the minumum association distance to the targes 
                targets_ones_axis_min[x*max_side:min([(x+1)*max_side,s_len])] = torch.minimum(
                    targets_ones_axis_min[x*max_side:min([(x+1)*max_side,s_len])],
                    mins
                )
        
        sources_indexs = sources_zero_axis_index.cpu().numpy().astype(int)
        targets_indexs = targets_ones_axis_index.cpu().numpy().astype(int)

        # Doing this on torch should speed this up
        # Check for matches (circular symetric minimum index)
        assoc = np.ones((targets_indexs.shape[0],5))*np.inf
        for x in range(targets_indexs.shape[0]):
            # Check for symetric minimum index 
            if sources_indexs[targets_indexs[x]] == x: 
                # Save made association
                assoc[x] = [x,targets_indexs[x],targets_ones_axis_min[x].cpu().numpy(), target_img_index, source_img_indexs]
            else:
                # Save not made association
                assoc[x] = [np.inf,np.inf,targets_ones_axis_min[x].cpu().numpy(),np.inf,np.inf]

        return assoc

    def predict(self, t_patches: torch.tensor, 
                t_masks: List[np.ndarray] = [None]): 
        
        # TODO weights are very slow I should fix this.

        t_patches = t_patches[0]
        
        self.nn_object.reset()
        self.nominal_patches = torch.cat([_[1] for _ in self.channels if _[1].size(0) >= self.min_channel_length], dim=0)[:,0,:]

        # Generate positional stds for positional mhalehnobis distances 
        # Uses torch.fiass 
        self.nn_object.add(self.nominal_patches)
        dist, ind = self.nn_object.search(torch.permute(t_patches,(1,0)),1)

        pos_weights = np.zeros_like(dist)

        # Positional consitency scoring 
        start = time.time()
        if self.pos_weight > 0.0:
            pos_mean = np.concatenate([np.repeat(np.mean([list(np.unravel_index(int(pos),
                                    (self.image_size[0]//self.scale,self.image_size[1]//self.scale))) 
                                    for pos in _[2]], axis=0, keepdims=True),len(_[1]),axis=0)
                                        for _ in self.channels if _[1].size(0) >= self.min_channel_length],axis=0)/np.sqrt(t_patches.size(1))
            pos_std = np.concatenate([np.repeat(np.std([list(np.unravel_index(int(pos),
                            (self.image_size[0]//self.scale,self.image_size[1]//self.scale))) 
                            for pos in _[2]], axis=0, keepdims=True),len(_[1]),axis=0)
                                for _ in self.channels if _[1].size(0) >= self.min_channel_length],axis=0)/np.sqrt(t_patches.size(1))
            for i in range(dist.shape[0]):
                pos = np.array(np.unravel_index(int(i),(self.image_size[0]//self.scale,self.image_size[1]//self.scale)))/np.sqrt(t_patches.size(1))
                scaling_factor = np.exp(-np.sqrt(np.sum(np.square(pos_std[ind[i],:]))))*self.pos_weight
                pos_weights[i] += np.sqrt(np.sum(np.square(pos-pos_mean[ind[i],:])))*scaling_factor

        dist = dist + pos_weights

        dist = np.resize(dist[:,0], new_shape=(self.image_size[0]//self.scale,self.image_size[1]//self.scale))
        dist = dist.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        scores = [gaussian_filter(dist,self.filter_size)]
        scores  = np.array(scores).flatten()

        if not t_masks[0] is None: 
            t_masks = [(mask[:,:,0]/255.).astype(int) for mask in t_masks]
            t_masks = np.array(t_masks).flatten()

        return scores, t_masks

    def check_if_nominal(self, index, num_patches, mask):
        if mask is None:
            return 0
        self.scale = int(np.sqrt(mask.shape[0]*mask.shape[1]//int(num_patches)))
        index = np.unravel_index(int(index),(mask.shape[0]//self.scale,mask.shape[1]//self.scale)) * self.scale
        return int(np.sum(mask[index[0]:index[0]+self.scale,index[1]:index[1]+self.scale]) == 0)

    def step(self, image: np.ndarray, mask: np.ndarray):
        # Start point 
        score = None
        scores_ = None
        features_start = time.time()
        # generate descriptors for patches in image 
        patches =  self.fd_gen.generate_descriptors([image], quite=True)
        if self.mode == 'testing': self.feature_extraction_time.append(abs(time.time()-features_start))
        num_p = patches.size(2)

        # Generate Nominal Labels
        scale = int(np.sqrt(mask.shape[0]*mask.shape[1]//int(num_p)))
        nominal_labels = torch.reshape(torch.from_numpy(block_reduce(mask.astype(np.float64),block_size=scale)), (1,num_p)).numpy()

        # If we have no channels we are at startup
        if len(self.channels) == 0:
            # Do startup with sequence (add all patches to newly generated channels)
            for p_i in range(patches.size(2)):
                self.channels.append([self.time_to_live, torch.unsqueeze(patches[:,:,p_i],dim=0), [p_i], [int(nominal_labels[0,p_i] == 0)]])
                self.channel_seeds.append(patches[:,:,p_i].cuda())
        else:
            # Upate at the update speed or every frame early on to get suitable channels 
            if self.count % self.update_interval == 0 or self.count <= 2*self.min_channel_length:
                # Decriment TTL and remove stale nominal channels
                start_time_channel = time.time()
                for x in range(len(self.channels)):
                    self.channels[len(self.channels)-x-1][0] -= 1
                    if self.channels[len(self.channels)-x-1][0] <= 0:
                        del self.channels[len(self.channels)-x-1]
                        del self.channel_seeds[len(self.channels)-x-1]
                # Predict and do associations
                channel_seeds_cat = torch.cat(self.channel_seeds).transpose(1,0)
                # Generate associations between new patcehs and existing channel seeds 
                assoc = self.gen_assoc(channel_seeds_cat, patches[0].cuda(), 0, 0)
                added_count = 0
                associated_count = 0

                # For each association
                for p_i in range(assoc.shape[0]):
                    # If it is associated to a channel add it to that channel 
                    if assoc[p_i,0] <= patches.size(2):
                        self.channels[int(assoc[p_i,1])][1] = torch.cat([self.channels[int(assoc[p_i,1])][1], torch.unsqueeze(patches[:,:,int(p_i)],dim=0)])
                        self.channels[int(assoc[p_i,1])][0] = self.time_to_live
                        self.channels[int(assoc[p_i,1])][2] += [p_i]
                        self.channels[int(assoc[p_i,1])][3] += [int(nominal_labels[0,p_i] == 0)]
                        associated_count += 1 
                    # If it is not associated to a channel create a new channel with that patch as it's seed 
                    else:
                        self.channels.append([self.time_to_live, torch.unsqueeze(patches[:,:,int(p_i)],dim=0), [p_i], [int(nominal_labels[0,p_i] == 0)]])
                        self.channel_seeds.append(patches[:,:,int(p_i)].cuda())
                        added_count += 1 
                if self.mode == 'testing': self.update_speed.append(abs(time.time()-start_time_channel))
            
            # Generate anomaly predictions 
            inferance_start = time.time()
            if self.count > self.min_channel_length*2:
                scores_, t_masks_ = self.predict(patches, [mask])
                if self.mode == 'testing':
                    self.inferance_speed.append(abs(time.time()-inferance_start)) 
                    self.scores.append(scores_)
                    self.t_masks.append(t_masks_)
                    self.image_wise_predictions.append(np.max(self.scores[-1]))
                    self.image_wise_actual.append(np.max(self.t_masks[-1]))

        self.count += 1
        
        if self.mode == 'testing':
            channel_p = [_[3] for _ in self.channels if _[1].size(0) >= self.min_channel_length]
            if len(channel_p) > 0 :
                channel_p = np.mean([item for sublist in channel_p for item in sublist])
                self.channel_percision.append(channel_p)

            self.num_channels.append(len(self.channels))
            self.pixel_wise_AUROC.append(score)

            for log_ in [self.scores, self.t_masks, self.inferance_speed, self.update_speed, self.image_wise_predictions, 
                        self.image_wise_actual, self.pixel_wise_AUROC, self.channel_percision, self.feature_extraction_time]:
                if len(log_) < self.count:
                    log_.append(None)

        if not scores_ is None: scores_ = np.resize(scores_, (self.image_size[0],self.image_size[1]))  
        return scores_, len(self.channels), patches

    def save_run_data(self,seed:int,experiment_dir:str='Experiments',experiment_name: str='Test'):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        directory = os.path.join(os.path.dirname(__file__), experiment_dir,sha,str(seed),experiment_name)
        if not os.path.exists(directory):
            os.makedirs(directory)  
        # Save self (the whole class tbh in a pickle file)
        save_dict = {
            'quite' : self.quite,
            'image_size' : self.image_size,
            'filter_size' : self.filter_size,
            'min_channel_length' : self.min_channel_length,
            'time_to_live' : self.time_to_live,
            'update_interval' : self.update_interval
        }

        with open(os.path.join(directory,'config.json'), 'w') as f:
            json.dump(save_dict, f)


        with open(os.path.join(directory,'results.csv'), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(['average','data_feild','results'])
            writer.writerow([np.median([_ for _ in self.inferance_speed if not _ is None]), 'inferance_speed']+self.inferance_speed)
            writer.writerow([np.median([_ for _ in self.update_speed if not _ is None]), 'update_speed']+self.update_speed)
            writer.writerow([np.mean([_ for _ in self.image_wise_predictions if not _ is None]), 'image_wise_predictions']+self.image_wise_predictions)
            writer.writerow([np.mean([_ for _ in self.image_wise_actual if not _ is None]), 'image_wise_actual']+self.image_wise_actual)
            writer.writerow([roc_auc_score(np.concatenate([_ for _ in self.t_masks if not _ is None]), np.concatenate([_ for _ in self.scores if not _ is None])), 'pixel_wise_AUROC']+self.pixel_wise_AUROC)
            writer.writerow([np.mean([_ for _ in self.channel_percision if not _ is None]), 'channel_percision']+self.channel_percision)
            writer.writerow([np.mean([_ for _ in self.num_channels if not _ is None]), 'num_channels']+self.num_channels)
            writer.writerow([np.median([_ for _ in self.feature_extraction_time if not _ is None]), 'feature_extraction_time']+self.feature_extraction_time)
            writer.writerow([roc_auc_score(np.array([_ for _ in self.image_wise_actual if not _ is None]), np.array([_ for _ in self.image_wise_predictions if not _ is None])), 
                             'REAL iamgewise AUROC'])

def test_online(ttl, minl, pos_w, seed, data_dir, exp_name, corr, tur):
    # Test things with a basic config
    super_seed(seed)
    class_names = [x for x in os.listdir(data_dir) if not '.' in x]
    class_names.sort()
    return_nodes = {
        'layer1.0.relu_2': 'Level_1',
        'layer1.1.relu_2': 'Level_2',
        'layer1.2.relu_2': 'Level_3',
        'layer2.0.relu_2': 'Level_4',
        'layer2.1.relu_2': 'Level_5',
        'layer2.2.relu_2': 'Level_6',
        'layer2.3.relu_2': 'Level_7',
        'layer3.1.relu_2': 'Level_8',
        'layer3.2.relu_2': 'Level_9',
        'layer3.3.relu_2': 'Level_10',
        'layer3.4.relu_2': 'Level_11',
        'layer3.5.relu_2': 'Level_12',
        'layer4.0.relu_2': 'Level_13'
        }

    model = load_wide_resnet_50(return_nodes=return_nodes, verbose=False)

    for class_name in class_names:
        print(class_name)
        super_seed(seed)
        images, masks = load_corrupted_data(class_name=class_name, 
                                                            data_dir=data_dir,
                                                            num_corrupted=9999)

        test_InReaCh = OnlineInReaCh(sample_image=images[0], model=model, min_channel_length=minl, time_to_live=ttl, pos_weight=pos_w, quite=False)

        for i in tqdm.tqdm(range(len(images)), ncols=100):
            temp = time.time()
            scores, num_channels, patches = test_InReaCh.step(images[i], masks[i])          
            if not scores is None: 
                if not os.path.exists('qual/'+exp_name+'/'+class_name+'/'):
                    os.makedirs('qual/'+exp_name+'/'+class_name+'/')
                cv2.imwrite('qual/'+exp_name+'/'+class_name+'/'+str(i)+'.png', visualize_confidence(images[i],scores,masks[i]))
        
        test_InReaCh.save_run_data(seed, experiment_name=exp_name+'/'+class_name)
    exit()

def test_pretrained(ttl, minl, pos_w, seed, data_dir, exp_name, corr, tur):
    # Test things with a basic config
    super_seed(seed)
    class_names = [x for x in os.listdir(data_dir) if not '.' in x]
    class_names.sort()
    return_nodes = {
        'layer1.0.relu_2': 'Level_1',
        'layer1.1.relu_2': 'Level_2',
        'layer1.2.relu_2': 'Level_3',
        'layer2.0.relu_2': 'Level_4',
        'layer2.1.relu_2': 'Level_5',
        'layer2.2.relu_2': 'Level_6',
        'layer2.3.relu_2': 'Level_7',
        'layer3.1.relu_2': 'Level_8',
        'layer3.2.relu_2': 'Level_9',
        'layer3.3.relu_2': 'Level_10',
        'layer3.4.relu_2': 'Level_11',
        'layer3.5.relu_2': 'Level_12',
        'layer4.0.relu_2': 'Level_13'
        }

    model = load_wide_resnet_50(return_nodes=return_nodes, verbose=False)

    for class_name in class_names:
        print(class_name)
        super_seed(seed)
        images, masks = load_corrupted_data(class_name=class_name, 
                                                            data_dir=data_dir,
                                                            num_corrupted=corr)

        test_InReaCh = OnlineInReaCh(sample_image=images[0], model=model, min_channel_length=minl, time_to_live=ttl, pos_weight=pos_w, quite=False)
        print('Training...')
        for i in tqdm.tqdm(range(len(images)), ncols=100):
            temp = time.time()
            scores, num_channels, patches = test_InReaCh.step(images[i], masks[i])   
            if not scores is None: 
                if not os.path.exists('qual/'+exp_name+'/'+class_name+'/'):
                    os.makedirs('qual/'+exp_name+'/'+class_name+'/')
                cv2.imwrite('qual/'+exp_name+'/'+class_name+'/_train'+str(i)+'.png', visualize_confidence(images[i],scores,masks[i]))    

        # Reset O-InReaCh for logging
        test_InReaCh.update_interval = tur 
        test_InReaCh.scores = []
        test_InReaCh.t_masks = []
        test_InReaCh.image_wise_actual = []
        test_InReaCh.pixel_wise_AUROC = []
        test_InReaCh.channel_percision = []
        test_InReaCh.image_wise_predictions = []  

        images, masks = load_corrupted_data(class_name=class_name, 
                                                            data_dir=data_dir,
                                                            num_nominal=0,
                                                            num_corrupted=9999)

        for i in tqdm.tqdm(range(len(images)), ncols=100):
            scores, num_channels, patches = test_InReaCh.step(images[i], masks[i])          
            if not scores is None: 
                if not os.path.exists('qual/'+exp_name+'/'+class_name+'/'):
                    os.makedirs('qual/'+exp_name+'/'+class_name+'/')
                cv2.imwrite('qual/'+exp_name+'/'+class_name+'/_test'+str(i)+'.png', visualize_confidence(images[i],scores,masks[i]))

        test_InReaCh.save_run_data(seed, experiment_name=exp_name+'/'+class_name)
    exit()

if __name__ == '__main__':

    # ArgParse
    parser = argparse.ArgumentParser(description='Online InReaCh on MVTec-like dataset.')
    parser.add_argument('-ttl',      type=int, dest='ttl', help='Time to live', default=15)
    parser.add_argument('-tur',      type=int, dest='tur', help='Test Update Rate', default=99999999)
    parser.add_argument('-minl',     type=int, dest='minl', help='Minimum Channel Length', default=3)
    parser.add_argument('-pos_w',     type=float, dest='pos_w', help='Posistional Consistentcy Weighting For Prediction', default=0.1)
    parser.add_argument('-seed',     type=int, dest='seed', help='Random Super Seed', default=112358)
    parser.add_argument('-data_dir', dest='data_dir', help='Root MVTec-like Directory', default='data/mvtec_anomaly_detection/')
    parser.add_argument('-n',        dest='exp_name', help='Experiment Name For Logging', default='Generic')
    parser.add_argument('-corr',     dest='corr', help='Number of Corrupted images to add (Only effects pre-trained testing, online testing always uses all corruptions.)', default=0)
    parser.add_argument('--pretrain',dest='t_type', action='store_const', const=test_pretrained, default=test_online, help='This flag converts testing configuration to pre-trined rather than the default of online.')


    args = parser.parse_args()

    print(args.exp_name)

    args.t_type(
        args.ttl,
        args.minl,
        args.pos_w,
        args.seed,
        args.data_dir,
        args.exp_name,
        args.corr,
        args.tur
    )

