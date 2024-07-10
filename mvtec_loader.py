import cv2 
import numpy as np
import os
import random
import itertools

def load_corrupted_data(class_name: str, 
                        data_dir: str, 
                        num_corrupted: int, 
                        num_nominal: int = 99999999, 
                        size: tuple = (256,256), 
                        crop_size: tuple = (224,224)):
    train_images = load_training_data(class_name, data_dir, size=size, crop_size=crop_size)
    test_images, test_masks, corr_types = load_testing_data(class_name, data_dir, size=size, crop_size=crop_size)
    
    images = test_images[:num_corrupted] + train_images[:num_nominal]
    masks  = test_masks[:num_corrupted] + [np.zeros_like(test_masks[0]) for x in range(len(train_images))][:num_nominal]
    #ziped = list(zip(images, masks)) # TODO uncomment this shuffle it was to test for all anomalies first in online setting 
    #random.shuffle(ziped)
    #images, masks = zip(*ziped)
    return images, masks

def load_testing_data(class_name: str, 
                      data_dir: str, 
                      size: tuple = (256,256), 
                      crop_size: tuple = (224,224)):
    assert class_name in os.listdir(data_dir)
    img_dir = data_dir+class_name+'/test/'
    ann_dir = data_dir+class_name+'/ground_truth/'
    test_images = []
    test_truths = []
    test_class = []
    x = int(size[0]/2- crop_size[0]/2)
    y = int(size[1]/2- crop_size[1]/2)
    for directory in os.listdir(img_dir):
        for filename in os.listdir(img_dir+directory+'/'):
            test_images.append(cv2.resize(cv2.imread(img_dir+directory+'/'+filename),size)[x:x+crop_size[0],y:y+crop_size[1]])
            if directory != 'good':
                test_truths.append(cv2.resize(cv2.imread(ann_dir+directory+'/'+filename[:-4]+'_mask.png'),size)[x:x+crop_size[0],y:y+crop_size[1]])
            else:
                test_truths.append(np.zeros_like(test_images[-1]))
            test_class.append(directory)

    ziped = list(zip(test_images, test_truths, test_class))
    random.shuffle(ziped)
    test_images, test_truths, test_class = zip(*ziped)

    return list(test_images), list(test_truths), list(test_class)

def load_training_data(class_name: str, 
                      data_dir: str, 
                      size: tuple = (224,224), 
                      crop_size: tuple = (224,224)):
    assert class_name in os.listdir(data_dir)
    train_images = []
    dir = data_dir+class_name+'/train/good/'
    x = int(size[0]/2- crop_size[0]/2)
    y = int(size[1]/2- crop_size[1]/2)
    for filename in os.listdir(dir):
        train_images.append(cv2.resize(cv2.imread(dir+filename),size)[x:x+crop_size[0],y:y+crop_size[1]])
    return train_images

def load_UCSD_dataset(data_dir: str,
                      size: tuple = (360,240),
                      crop_size: tuple = (352,224),
                      diff_mode = False):
    # We only consider UCSDped2 since UCSDped1 has missing ground truths for evaluation.... 
    # Center crop to 224x352 
    test_dir  = 'UCSD_ped/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/'
    train_dir = 'UCSD_ped/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/'



    x = int(size[1]/2- crop_size[1]/2)
    y = int(size[0]/2- crop_size[0]/2)

    image_sequences = []
    mask_sequences = []

    # Load testing data 
    for dir_case in [test_dir, train_dir]:
        for directory in os.listdir(data_dir + dir_case):
            if (directory[0:4] == 'Test' or directory[0:5] == 'Train')  and directory[-1] != 't':
                image_sequence = []
                mask_sequence = []
                filenames = [x for x in os.listdir(data_dir + dir_case+directory+'/') if x[-4:] == '.tif']
                filenames.sort()
                if os.path.exists(data_dir + dir_case+directory+'_gt/'):
                    mask_names = [x for x in os.listdir(data_dir + dir_case+directory+'_gt/') if x[-4:] == '.bmp']
                    mask_names.sort()
                else: 
                    mask_names = None

                for k in range(len(filenames)):
                    
                    image_sequence.append(cv2.resize(cv2.imread(data_dir + dir_case+directory+'/'+filenames[k]),size)[x:x+crop_size[1],y:y+crop_size[0]])
                    #print(image_sequence[-1].shape, cv2.imread(data_dir + dir_case+directory+'/'+filenames[k]).shape)
                    if not mask_names is None :
                        mask_sequence.append(cv2.resize(cv2.imread(data_dir + dir_case+directory+'_gt/'+mask_names[k]),size)[x:x+crop_size[1],y:y+crop_size[0]])
                    else:
                        mask_sequence.append(np.zeros_like(image_sequence[-1]))

                if diff_mode:
                    diff_sequnce = []
                    for k in range(0,len(image_sequence)-1):
                        diff_sequnce.append(image_sequence[k].astype(float)-image_sequence[k+1].astype(float))
                    image_sequences.append(diff_sequnce)
                    mask_sequences.append(mask_sequence[:-1])               
                else:
                    image_sequences.append(image_sequence)
                    mask_sequences.append(mask_sequence)

    ziped = list(zip(image_sequences, mask_sequences))
    random.shuffle(ziped)
    image_sequences, mask_sequences = zip(*ziped)

    image_sequences = list(itertools.chain.from_iterable(image_sequences))
    mask_sequences = list(itertools.chain.from_iterable(mask_sequences))

    #for x in range(len(image_sequences)):
    #    cv2.imshow('test', image_sequences[x])
    #    cv2.imshow('mask', mask_sequences[x])
    #    cv2.waitKey(10)


    return image_sequences, mask_sequences


#load_UCSD_dataset(data_dir='data/')