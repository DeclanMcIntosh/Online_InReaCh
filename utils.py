import torch
import numpy as np
import random
from typing import List
import cv2
import tqdm


def measure_distances(features_a, features_b):
    distances = torch.cdist(torch.permute(features_a,[1,0]),torch.permute(features_b,[1,0]))
    return distances

def super_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)


def visualize_confidence(img, pred, truth=None, display_size=(512,512)):

    

    pred = np.expand_dims(pred,2)
    pred = np.repeat(pred,3,axis=2)    
    pred = np.exp(pred)

    pred = (pred - np.min(pred))
    pred = pred/3
    pred = np.clip(pred, 0, 1)
    
    score_img = cv2.applyColorMap((pred*255).astype(np.uint8), cv2.COLORMAP_JET)
    

    img = (img.astype(np.float32)*(1-np.sqrt(pred))+score_img.astype(np.float32)*np.sqrt(pred)).astype(np.uint8)

    img = cv2.resize(img, (512,512))

    if not truth is None:
        truth = cv2.resize(truth, (512,512))
        edged = cv2.Canny(truth, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

    return img