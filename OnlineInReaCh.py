import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from typing import List, Tuple, Dict, Optional
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
import numpy as np
import torch

from FeatureDescriptors import *
from utils import *
from model import *
from mvtec_loader import *


class OnlineInReaCh:
    def __init__(self,
                 sample_image: np.ndarray,
                 model: torch.nn.Module,
                 min_channel_length: int = 2,
                 filter_size: float = 13,
                 time_to_live: int = 3,
                 pos_weight: float = 0.0,
                 update_interval: int = 1,
                 quite: bool = False,
                 mode: str = 'testing',
                 **kwargs) -> None:
        """
        Initialize the OnlineInReaCh class.

        Args:
            sample_image (np.ndarray): A sample image to determine image size.
            model (torch.nn.Module): The feature extraction model.
            min_channel_length (int): Minimum number of patches required for a channel to be valid.
            filter_size (float): Size of the Gaussian filter for smoothing.
            time_to_live (int): Time-to-live for channels before they are removed.
            pos_weight (float): Weight for positional consistency in predictions.
            update_interval (int): Interval for updating channels.
            quite (bool): If True, suppress logging output.
            mode (str): Mode of operation ('testing' or 'training').
            **kwargs: Additional arguments for feature descriptor initialization.
        """
        self.quite = quite
        self.image_size = tuple(sample_image.shape)
        self.model = model
        self.pos_weight = pos_weight
        self.filter_size = filter_size
        self.min_channel_length = min_channel_length
        self.time_to_live = time_to_live
        self.update_interval = update_interval
        self.mode = mode

        # Setup feature extraction
        self.fd_gen = Feautre_Descriptor(model=model, image_size=self.image_size, **kwargs)
        self.scale = 4  # TODO: Avoid hardcoding this value

        # Setup channels
        self.channels: List[List] = []
        self.channel_seeds: List[torch.Tensor] = []
        self.count = 0

        # Logging
        self.scores: List = []
        self.t_masks: List = []
        self.inferance_speed: List = []
        self.update_speed: List = []
        self.image_wise_predictions: List = []
        self.image_wise_actual: List = []
        self.pixel_wise_AUROC: List = []
        self.channel_percision: List = []
        self.num_channels: List = []
        self.feature_extraction_time: List = []

        # Faiss
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        self.nn_object = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 1024, flat_config)

    def gen_assoc(self,
                  targets: torch.Tensor,
                  sources: torch.Tensor,
                  target_img_index: int,
                  source_img_indexs: int) -> np.ndarray:
        """
        Generate associations between targets and sources.

        Args:
            targets (torch.Tensor): Target feature descriptors.
            sources (torch.Tensor): Source feature descriptors.
            target_img_index (int): Index of the target image.
            source_img_indexs (int): Index of the source images.

        Returns:
            np.ndarray: Associations between targets and sources.
        """
        t_len = targets.size()[1]
        s_len = sources.size()[1]
        sources_zero_axis_min = torch.from_numpy(np.ones(shape=(t_len)) * np.inf).cuda()
        sources_zero_axis_index = torch.from_numpy(np.zeros(shape=(t_len))).cuda()
        targets_ones_axis_min = torch.from_numpy(np.ones(shape=(s_len)) * np.inf).cuda()
        targets_ones_axis_index = torch.from_numpy(np.zeros(shape=(s_len))).cuda()

        # Handle GPU memory constraints
        aval_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        max_side = int(np.floor(np.sqrt(aval_mem // 32)))
        for x in range(int(np.ceil(s_len / max_side))):
            for y in range(int(np.ceil(t_len / max_side))):
                distances = measure_distances(
                    sources[:, x * max_side:min([(x + 1) * max_side, s_len])],
                    targets[:, y * max_side:min([(y + 1) * max_side, t_len])]
                )

                # Update minimum distances and indices
                mins, args = torch.min(distances, axis=0)
                sources_zero_axis_index[y * max_side:min([(y + 1) * max_side, t_len])] = torch.where(
                    sources_zero_axis_min[y * max_side:min([(y + 1) * max_side, t_len])] >= mins,
                    args + x * max_side,
                    sources_zero_axis_index[y * max_side:min([(y + 1) * max_side, t_len])]
                )
                sources_zero_axis_min[y * max_side:min([(y + 1) * max_side, t_len])] = torch.minimum(
                    sources_zero_axis_min[y * max_side:min([(y + 1) * max_side, t_len])],
                    mins
                )

                mins, args = torch.min(distances, axis=1)
                targets_ones_axis_index[x * max_side:min([(x + 1) * max_side, s_len])] = torch.where(
                    targets_ones_axis_min[x * max_side:min([(x + 1) * max_side, s_len])] >= mins,
                    args + y * max_side,
                    targets_ones_axis_index[x * max_side:min([(x + 1) * max_side, s_len])]
                )
                targets_ones_axis_min[x * max_side:min([(x + 1) * max_side, s_len])] = torch.minimum(
                    targets_ones_axis_min[x * max_side:min([(x + 1) * max_side, s_len])],
                    mins
                )

        sources_indexs = sources_zero_axis_index.cpu().numpy().astype(int)
        targets_indexs = targets_ones_axis_index.cpu().numpy().astype(int)

        # Check for symmetric matches
        assoc = np.ones((targets_indexs.shape[0], 5)) * np.inf
        for x in range(targets_indexs.shape[0]):
            if sources_indexs[targets_indexs[x]] == x:
                assoc[x] = [x, targets_indexs[x], targets_ones_axis_min[x].cpu().numpy(), target_img_index, source_img_indexs]
            else:
                assoc[x] = [np.inf, np.inf, targets_ones_axis_min[x].cpu().numpy(), np.inf, np.inf]

        return assoc

    def predict(self,
                t_patches: torch.Tensor,
                t_masks: List[np.ndarray] = [None]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores for patches.

        Args:
            t_patches (torch.Tensor): Target patches.
            t_masks (List[np.ndarray]): Ground truth masks.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Anomaly scores and ground truth masks.
        """
        # TODO: Optimize weights computation
        t_patches = t_patches[0]
        self.nn_object.reset()
        self.nominal_patches = torch.cat(
            [_[1] for _ in self.channels if _[1].size(0) >= self.min_channel_length], dim=0
        )[:, 0, :]

        self.nn_object.add(self.nominal_patches)
        dist, ind = self.nn_object.search(torch.permute(t_patches, (1, 0)), 1)

        pos_weights = np.zeros_like(dist)

        # Positional consistency scoring
        if self.pos_weight > 0.0:
            pos_mean = np.concatenate([
                np.repeat(
                    np.mean([
                        list(np.unravel_index(int(pos), (self.image_size[0] // self.scale, self.image_size[1] // self.scale)))
                        for pos in _[2]
                    ], axis=0, keepdims=True),
                    len(_[1]), axis=0
                )
                for _ in self.channels if _[1].size(0) >= self.min_channel_length
            ], axis=0) / np.sqrt(t_patches.size(1))

            pos_std = np.concatenate([
                np.repeat(
                    np.std([
                        list(np.unravel_index(int(pos), (self.image_size[0] // self.scale, self.image_size[1] // self.scale)))
                        for pos in _[2]
                    ], axis=0, keepdims=True),
                    len(_[1]), axis=0
                )
                for _ in self.channels if _[1].size(0) >= self.min_channel_length
            ], axis=0) / np.sqrt(t_patches.size(1))

            for i in range(dist.shape[0]):
                pos = np.array(np.unravel_index(int(i), (self.image_size[0] // self.scale, self.image_size[1] // self.scale))) / np.sqrt(t_patches.size(1))
                scaling_factor = np.exp(-np.sqrt(np.sum(np.square(pos_std[ind[i], :])))) * self.pos_weight
                pos_weights[i] += np.sqrt(np.sum(np.square(pos - pos_mean[ind[i], :]))) * scaling_factor

        dist = dist + pos_weights
        dist = np.resize(dist[:, 0], new_shape=(self.image_size[0] // self.scale, self.image_size[1] // self.scale))
        dist = dist.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        scores = [gaussian_filter(dist, self.filter_size)]
        scores = np.array(scores).flatten()

        if t_masks[0] is not None:
            t_masks = [(mask[:, :, 0] / 255.).astype(int) for mask in t_masks]
            t_masks = np.array(t_masks).flatten()

        return scores, t_masks
    
    def check_if_nominal(self, index: int, num_patches: int, mask: Optional[np.ndarray]) -> int:
        """
        Check if a patch is nominal (non-anomalous) based on the mask.

        Args:
            index (int): The index of the patch to check.
            num_patches (int): The total number of patches.
            mask (Optional[np.ndarray]): The ground truth mask for the image.

        Returns:
            int: 1 if the patch is nominal, 0 otherwise.
        """
        if mask is None:
            return 0

        # Calculate the scale factor based on the mask size and number of patches
        self.scale = int(np.sqrt(mask.shape[0] * mask.shape[1] // num_patches))

        # Determine the patch's position in the mask
        index = np.unravel_index(index, (mask.shape[0] // self.scale, mask.shape[1] // self.scale)) * self.scale

        # Check if the patch is nominal (sum of mask values in the patch is zero)
        return int(np.sum(mask[index[0]:index[0] + self.scale, index[1]:index[1] + self.scale]) == 0)

    def save_run_data(self, seed: int, experiment_dir: str = 'Experiments', experiment_name: str = 'Test', save_imgs: bool = False) -> None:
        """
        Save the run data and configuration to a directory.

        Args:
            seed (int): The random seed used for the experiment.
            experiment_dir (str): The base directory for saving experiments.
            experiment_name (str): The name of the experiment.
        """
        # Get the current Git commit hash
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        # Create the directory for saving the experiment
        directory = os.path.join(os.path.dirname(__file__), experiment_dir, experiment_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Prepare the data to save
        save_dict = {
            'quite': self.quite,
            'image_size': self.image_size,
            'filter_size': self.filter_size,
            'min_channel_length': self.min_channel_length,
            'time_to_live': self.time_to_live,
            'update_interval': self.update_interval,
            'inferance_speed': np.median([_ for _ in self.inferance_speed if _ is not None]),
            'update_speed': np.median([_ for _ in self.update_speed if _ is not None]),
            'image_wise_predictions': np.mean([_ for _ in self.image_wise_predictions if _ is not None]),
            'image_wise_actual': np.mean([_ for _ in self.image_wise_actual if _ is not None]),
            'pixel_wise_AUROC': roc_auc_score(
                np.concatenate([_ for _ in self.t_masks if _ is not None]),
                np.concatenate([_ for _ in self.scores if _ is not None])
            ),
            'channel_percision': np.mean([_ for _ in self.channel_percision if _ is not None]),
            'num_channels': np.mean([_ for _ in self.num_channels if _ is not None]),
            'feature_extraction_time': np.median([_ for _ in self.feature_extraction_time if _ is not None]),
            'REAL_imagewise_AUROC': roc_auc_score(
                np.array([_ for _ in self.image_wise_actual if _ is not None]),
                np.array([_ for _ in self.image_wise_predictions if _ is not None])
            ),
            'git_commit': sha,
            'seed': seed,
        }
        save_dict = make_json_serializable(save_dict)
        
        # Print the saved data
        print(f"Saving the following run data to {directory}")
        for key, value in save_dict.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")

        # Save the configuration as a JSON file
        with open(os.path.join(directory, 'config_and_results.json'), 'w') as f:
            json.dump(save_dict, f)

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


def test_online(ttl: int, minl: int, pos_w: float, seed: int, data_dir: str, exp_name: str, corr: int, tur: int, save_images: bool) -> None:
    """
    Perform online testing with the specified configuration.

    Args:
        ttl (int): Time-to-live for channels.
        minl (int): Minimum channel length.
        pos_w (float): Positional consistency weighting for prediction.
        seed (int): Random seed for reproducibility.
        data_dir (str): Directory containing the dataset.
        exp_name (str): Name of the experiment for logging.
        corr (int): Number of corrupted images to add.
        tur (int): Test update rate.
    """
    super_seed(seed)
    class_names = [x for x in os.listdir(data_dir) if not '.' in x]
    class_names.sort()

    # Define the return nodes for the model
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

    # Load the model
    model = load_wide_resnet_50(return_nodes=return_nodes, verbose=False)

    for class_name in class_names:
        print(class_name)
        super_seed(seed)

        # Load corrupted data
        images, masks = load_corrupted_data(class_name=class_name, data_dir=data_dir, num_corrupted=9999)

        # Initialize OnlineInReaCh
        test_InReaCh = OnlineInReaCh(sample_image=images[0], model=model, min_channel_length=minl, time_to_live=ttl, pos_weight=pos_w, quite=False)

        # Process each image
        for i in tqdm.tqdm(range(len(images)), ncols=100):
            scores, num_channels, patches = test_InReaCh.step(images[i], masks[i])
            if scores is not None:
                output_dir = f'qual/{exp_name}/{class_name}/'
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(f'{output_dir}{i}.png', visualize_confidence(images[i], scores, masks[i]))

        # Save run data
        test_InReaCh.save_run_data(seed, experiment_name=f'{exp_name}/{class_name}', save_imgs=save_images)


def test_pretrained(ttl: int, minl: int, pos_w: float, seed: int, data_dir: str, exp_name: str, corr: int, tur: int, save_images: bool) -> None:
    """
    Perform testing with a pre-trained configuration.

    Args:
        ttl (int): Time-to-live for channels.
        minl (int): Minimum channel length.
        pos_w (float): Positional consistency weighting for prediction.
        seed (int): Random seed for reproducibility.
        data_dir (str): Directory containing the dataset.
        exp_name (str): Name of the experiment for logging.
        corr (int): Number of corrupted images to add.
        tur (int): Test update rate.
    """
    super_seed(seed)
    class_names = [x for x in os.listdir(data_dir) if not '.' in x]
    class_names.sort()

    # Define the return nodes for the model
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

    # Load the model
    model = load_wide_resnet_50(return_nodes=return_nodes, verbose=False)

    for class_name in class_names:
        print(class_name)
        super_seed(seed)

        # Load corrupted data for training
        images, masks = load_corrupted_data(class_name=class_name, data_dir=data_dir, num_corrupted=corr)

        # Initialize OnlineInReaCh
        test_InReaCh = OnlineInReaCh(sample_image=images[0], model=model, min_channel_length=minl, time_to_live=ttl, pos_weight=pos_w, quite=False)

        # Training phase
        print('Training...')
        for i in tqdm.tqdm(range(len(images)), ncols=100):
            scores, num_channels, patches = test_InReaCh.step(images[i], masks[i])
            if scores is not None:
                output_dir = f'qual/{exp_name}/{class_name}/'
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(f'{output_dir}_train{i}.png', visualize_confidence(images[i], scores, masks[i]))

        # Reset for testing
        test_InReaCh.update_interval = tur
        test_InReaCh.scores = []
        test_InReaCh.t_masks = []
        test_InReaCh.image_wise_actual = []
        test_InReaCh.pixel_wise_AUROC = []
        test_InReaCh.channel_percision = []
        test_InReaCh.image_wise_predictions = []

        # Load corrupted data for testing
        images, masks = load_corrupted_data(class_name=class_name, data_dir=data_dir, num_nominal=0, num_corrupted=9999)

        # Testing phase
        for i in tqdm.tqdm(range(len(images)), ncols=100):
            scores, num_channels, patches = test_InReaCh.step(images[i], masks[i])
            if scores is not None:
                output_dir = f'qual/{exp_name}/{class_name}/'
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(f'{output_dir}_test{i}.png', visualize_confidence(images[i], scores, masks[i]))

        # Save run data
        test_InReaCh.save_run_data(seed, experiment_name=f'{exp_name}/{class_name}', save_imgs=save_images)


def summarize_experiments_average(experiment_dir: str, quite=False) -> Dict[str, float]:
    """
    Summarize the results of all experiments in a directory and average metrics over all classes.

    Args:
        experiment_dir (str): The base directory containing experiment subdirectories.

    Returns:
        Dict[str, float]: A dictionary containing the averaged metrics across all classes.
    """
    metrics = {
        'Pixel-wise AUROC': [],
        'Image-wise AUROC': [],
        'Inference Speed (ms)': [],
        'Update Speed (ms)': [],
        'Number of Channels': [],
        'Channel Precision': []
    }

    # Traverse the experiment directory
    for root, dirs, files in os.walk(experiment_dir):
        for file in files:
            if file == 'config_and_results.json':
                config_path = os.path.join(root, file)
                try:
                    # Load the JSON file
                    with open(config_path, 'r') as f:
                        config = json.load(f)

                    # Append metrics to the lists
                    metrics['Pixel-wise AUROC'].append(config.get('pixel_wise_AUROC', None))
                    metrics['Image-wise AUROC'].append(config.get('REAL_imagewise_AUROC', None))
                    metrics['Inference Speed (ms)'].append(config.get('inferance_speed', None))
                    metrics['Update Speed (ms)'].append(config.get('update_speed', None))
                    metrics['Number of Channels'].append(config.get('num_channels', None))
                    metrics['Channel Precision'].append(config.get('channel_percision', None))
                except Exception as e:
                    print(f"Error reading {config_path}: {e}")

    # Compute averages for each metric
    averaged_metrics = {metric: sum(values) / len(values) if values else None for metric, values in metrics.items()}

    if not quite:
        # Print the averaged metrics
        print("Averaged Experiment Summary:")
        for metric, value in averaged_metrics.items():
            print(f"{metric}: {value:.3f}" if value is not None else f"{metric}: No data")

    return averaged_metrics


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
    parser.add_argument('--save_images', nargs='?', type=bool, default=False, help='Save images with predictions to disk.')
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
        args.tur,
        args.save_images,
    )

    summarize_experiments_average(os.path.join(os.path.dirname(__file__), 'Experiments', args.exp_name))
