import numpy as np
import os
import sys
sys.path.append('../')
import glob
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from models import FakeNet,VGGNet,DiscNet,AlexNet,MobileNetV2
import argparse

parser = argparse.ArgumentParser(description='Anti-forensic evaluation')
parser.add_argument('--imgs_dir', type=str, default='', help='input image directory.')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
parser.add_argument('--model_num', type=int, default=1, help='model to attack.') 
parser.add_argument('--ckpt_fakenet', type=str, default="../checkPoint/FakeNet/model.pth", help='path to FakeNet model.') 
parser.add_argument('--ckpt_vggnet', type=str, default="../checkPoint/VGGNet/model.pth", help='path to VGGNet model.')
parser.add_argument('--ckpt_discnet', type=str, default="../checkPoint/DiscNet/model.pth", help='path to DiscNet model.')
parser.add_argument('--ckpt_alexnet', type=str, default="../checkPoint/AlexNet/model.pth", help='path to AlexNet model.')
parser.add_argument('--ckpt_mobilenet', type=str, default="../checkPoint/MobNet/model.pth", help='path to MobileNetV2 model.')
parser.add_argument('--max_epsilon1', type=int, default=2, help='maximum perturbation at Y.')
parser.add_argument('--max_epsilon2', type=int, default=6, help='maximum perturbation at Cb.')
parser.add_argument('--max_epsilon3', type=int, default=6, help='maximum perturbation at Cr.')
parser.add_argument('--num_iter', type=int, default=10, help='number of iterations.')
parser.add_argument('--image_format', type=str, default='png', help='image format.')
parser.add_argument('--image_width', type=int, default=128, help='width of image.')
parser.add_argument('--image_height', type=int, default=128, help='height of image.')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imgs_dir = args.imgs_dir + 'src_m' + str(args.model_num) + '_eps_' + str(args.max_epsilon1) + \
           str(args.max_epsilon2) + str(args.max_epsilon3) +  '/' 

record_dir = args.imgs_dir + 'record/'

if not os.path.isdir(record_dir):
    os.mkdir(record_dir)
    

preprocess = transforms.ToTensor()


def get_file_list(dataset_dir):
    file_list = glob.glob(os.path.join(dataset_dir, '*.png')) + glob.glob(os.path.join(dataset_dir, '*.jpg'))
    file_list.sort()
    return file_list
    

def read_image(img_path):
    img = Image.open(img_path)
    img_tensor = preprocess(img)
    input_tensor = img_tensor.unsqueeze(0)
    return input_tensor
    

def save_records(acc, model_name='1', save_dir=record_dir):
    acc_record_path = save_dir + 'src_m' + str(args.model_num) + '_eps_' + str(args.max_epsilon1) + \
                       str(args.max_epsilon2) + str(args.max_epsilon3) + '_eval_m' + model_name + '.txt'
    print('Evaluated on model: %s, imgs_dir: %s, acc= %.3f' % (model_name, args.imgs_dir, acc))
    f = open(acc_record_path, 'w')
    print('Evaluated on model: %s, imgs_dir: %s, acc= %.3f' % (model_name, args.imgs_dir, acc), file=f)
    
# load model for attack
def load_model(model_path, net):
    "load pretrained model to network"
    model = net().to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))   
    return model

def get_acc(model_path, net, input_dir, gt_label):
    model =  load_model(model_path, net)
    model.eval()
    
    img_lists = get_file_list(input_dir)
    cnt = 0
    for img_idx, img_path in enumerate(img_lists):
        img = read_image(img_path)
        img = img.to(device)
        output = model(img)
        _, pred = torch.max(output, 1)
        if pred.data[0].cpu().numpy() == gt_label:
            cnt += 1
    acc = cnt / len(img_lists)
    return acc
           
    
# create label tensor
folder_name = args.imgs_dir
imgs_label = folder_name.split('/')[-2]   

if imgs_label.split('_')[-1] == 'real':
    img_label = 1
elif imgs_label.split('_')[-1] == 'fake':
    img_label = 0
else:
    raise ValueError('check folder name')


print('\n============================================================================')
print('Evaluating models on attacked face images from directory: %s'%(imgs_dir))
print('============================================================================\n')
#evaluate FakeNet
print('\n============================================================================')
print('Evaluating on model 1 ...')
acc = get_acc(args.ckpt_fakenet, FakeNet, imgs_dir, img_label)
save_records(acc, '1', record_dir)
print('============================================================================\n')
  
    
#evaluate VGGNet
print('\n============================================================================')
print('Evaluating on model 2 ...')
acc = get_acc(args.ckpt_vggnet, VGGNet, imgs_dir, img_label)
save_records(acc, '2', record_dir)
print('============================================================================\n')
  
    
#evaluate DiscNet
print('\n============================================================================')
print('Evaluating on model 3 ...')
acc = get_acc(args.ckpt_discnet, DiscNet, imgs_dir, img_label)
save_records(acc, '3', record_dir)
print('============================================================================\n')
    
    
#evaluate AlexNet
print('\n============================================================================')
print('Evaluating on model 4 ...')
acc = get_acc(args.ckpt_alexnet, AlexNet, imgs_dir, img_label)
save_records(acc, '4', record_dir)
print('============================================================================\n')


#evaluate MobileNetV2
print('\n============================================================================')
print('Evaluating on model 5 ...')
acc = get_acc(args.ckpt_mobilenet, MobileNetV2, imgs_dir, img_label)
save_records(acc, '5', record_dir)
print('============================================================================\n')    



