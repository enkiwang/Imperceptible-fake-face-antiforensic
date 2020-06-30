import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn


def get_file_list(dataset_dir):
    file_list = glob.glob(os.path.join(dataset_dir, '*.png')) + glob.glob(os.path.join(dataset_dir, '*.jpg'))
    file_list.sort()
    return file_list

def get_pair_list(imgs_path_org, imgs_path_distort, img_format='.png'):
    img_lists_distort = get_file_list(imgs_path_distort)
    img_lists_org = []
    for i in range(len(img_lists_distort)):
        img_name = img_lists_distort[i].split('/')[-1]
        img_name_no_format = img_name.split('.')[0]
        img_name_new = img_name_no_format + img_format 
        img_lists_org.append(os.path.join(imgs_path_org, img_name_new))
    return img_lists_org, img_lists_distort

def read_image_np(img_path):
    """
    read image, and normalize images within [0,1].
    """
    img = Image.open(img_path)
    return np.array(img) / 255.

def read_image(img_path):
    img = Image.open(img_path)
    img_tensor = preprocess(img)
    input_tensor = img_tensor.unsqueeze(0)
    return input_tensor

def read_image_pil(img_path):
    """
    read image, and normalize images within [0,1].
    """
    img = Image.open(img_path)
    return img

def save_image_pil(img_pil, img_name, save_dir, format='.png'):
    # img_pil: Pillow; img_name: 0; save_dir:faces/; format:.jpg
    img_save_path = save_dir + img_name + format
    img_pil.save(img_save_path, "PNG")

def read_sz_save(img_dir_input, img_dir_save):
    img_lists = get_file_list(img_dir_input)
    for img_idx, img_path in enumerate(img_lists):
        print('process %d img' %(img_idx+1))
        img = read_image_pil(img_path)
        img = img.resize((args.img_sz,args.img_sz),Image.LANCZOS)
        save_image_pil(img, str(img_idx+1), img_dir_save)

def save_image_tensor(img_save_path, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(img_save_path)
    

    
   