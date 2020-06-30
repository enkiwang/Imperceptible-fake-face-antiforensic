import numpy as np
import os
import sys
sys.path.append('../')
import glob
from PIL import Image
import random
import copy
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from models import FakeNet,VGGNet,DiscNet,AlexNet,MobileNetV2
import argparse

parser = argparse.ArgumentParser(description='Imperceptible fake face anti-forensics')
parser.add_argument('--input_dir', type=str, default='', help='input directory.')
parser.add_argument('--output_dir', type=str, default='', help='output directory.')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
parser.add_argument('--model_num', type=int, default=1, help='model to attack.') 
parser.add_argument('--ckpt_fakenet', type=str, default="../checkPoint/FakeNet/model.pth", help='path to FakeNet model.') 
parser.add_argument('--ckpt_vggnet', type=str, default="../checkPoint/VGGNet/model.pth", help='path to VGGNet model.')
parser.add_argument('--ckpt_discnet', type=str, default="../checkPoint/DiscNet/model.pth", help='path to DiscNet model.')
parser.add_argument('--ckpt_alexnet', type=str, default="../checkPoint/AlexNet/model.pth", help='path to AlexNet model.')
parser.add_argument('--ckpt_mobilenet', type=str, default="../checkPoint/MobNet/model.pth", help='path to MobileNetV2 model.')
parser.add_argument('--max_epsilon', type=int, default=4, help='maximum perturbation.')
parser.add_argument('--num_iter', type=int, default=10, help='number of iterations.')
parser.add_argument('--image_format', type=str, default='png', help='image format.')
parser.add_argument('--image_width', type=int, default=128, help='width of image.')
parser.add_argument('--image_height', type=int, default=128, help='height of image.')
parser.add_argument('--momentum', type=float, default=1.0, help='momentum.')
parser.add_argument('--max_epsilon1', type=int, default=2, help='maximum perturbation at Y.')
parser.add_argument('--max_epsilon2', type=int, default=6, help='maximum perturbation at Cb.')
parser.add_argument('--max_epsilon3', type=int, default=6, help='maximum perturbation at Cr.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_dir = args.output_dir + 'src_m' + str(args.model_num) + '_eps_' + str(args.max_epsilon1) + str(args.max_epsilon2) + str(args.max_epsilon3) + '/'
os.makedirs(output_dir, exist_ok=True)

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


def ycbcr2rgb_np(img):
    invA = np.array([[1.1644, 0.0, 1.5960], [1.1644, -0.3918, -0.8130], [1.1644, 2.0172, 0.0] ])
    img = img.astype(np.float)
    img[:,:,[1,2]] -= 128
    img[:,:,0] -= 16
    rgb = img.dot(invA.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.around(rgb) 


def rgb2ycbcr_np(img):
    #image as np.float, within range [0,255]
    A = np.array([[0.2568, 0.5041, 0.0979], [-0.1482, -0.2910, 0.4392], [0.4392, -0.3678, -0.0714]])
    ycbcr = img.dot(A.T)
    ycbcr[:,:,[1,2]] += 128
    ycbcr[:,:,0] += 16
    return ycbcr 


def read_image_ycbcr(img_path):
    img = Image.open(img_path)  
    img = np.array(img).astype('float') 
    img_ycbcr = rgb2ycbcr_np(img)   
    return img_ycbcr

def ycbcr_to_tensor(img_ycc):
    img_ycc = img_ycc.transpose(2,0,1) / 255. 
    img_ycc_tensor = torch.Tensor(img_ycc)
    return img_ycc_tensor.unsqueeze(0)


def ycbcr_to_rgb(img_ycc):
    img_ycc = img_ycc.squeeze(0)
    img_ycc = img_ycc.permute(1,2,0).view(-1,3).float()
    invA = torch.tensor([[1.164, 1.164, 1.164],
                        [0, -0.392, 2.0172],
                        [1.5960, -0.8130, 0]])
                      
    invb = torch.tensor([-16.0/255.0, -128.0/255.0, -128.0/255.0])
    invA, invb = invA.to(device), invb.to(device)
    img_ycc = (img_ycc + invb).mm(invA)
    img_ycc = img_ycc.view(args.image_height, args.image_width, 3)
    img_ycc = img_ycc.permute(2,0,1)
    img_ycc = img_ycc.unsqueeze(0)
    img_ycc = torch.clamp(img_ycc, min=0., max=1.)
    return img_ycc


def save_image(img_tensor, img_name, save_path):    
    img = img_tensor.squeeze(0)        
    img = img.detach().cpu().numpy()   
    img = img.transpose(1,2,0)
    img = np.clip(img * 255, 0, 255) 
    img = img.astype("uint8")
    img = Image.fromarray(img)
    img_save_path = save_path + img_name + '.png'
    img.save(img_save_path, "PNG")


# load model for attack
def load_model(model_path, net):
    "load pretrained model to network"
    model = net().to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))   
    return model

if args.model_num == 1:
    model = load_model(args.ckpt_fakenet, FakeNet)
if args.model_num == 2:
    model = load_model(args.ckpt_vggnet, VGGNet)
if args.model_num == 3:
    model = load_model(args.ckpt_discnet, DiscNet)
if args.model_num == 4:
    model = load_model(args.ckpt_alexnet, AlexNet)
if args.model_num == 5:
    model = load_model(args.ckpt_mobilenet, MobileNetV2) 

model.eval()


img_lists = get_file_list(args.input_dir)
# create label tensor
folder_name = args.input_dir
imgs_label = folder_name.split('/')[-2]

if imgs_label.split('_')[-1] == 'real':
    img_label_tensor = torch.Tensor([1]).long()
elif imgs_label.split('_')[-1] == 'fake':
    img_label_tensor = torch.Tensor([0]).long()
else:
    raise ValueError('check folder name')


def get_file_name(img_path, img_type=args.image_format):
    if img_type == 'png':
        img_name = img_path.split('.png')[-2].split('/')[-1]
    elif img_type == 'jpg':
        img_name = img_path.split('.jpg')[-2].split('/')[-1]
    else:
        raise ValueError('check image format.')
    return img_name
        

def attack(img_ycc, label, model, eps, mu=1.0, iterMax=args.num_iter):
    g_old = torch.zeros_like(img_ycc)
    img_ycc_org = copy.deepcopy(img_ycc.detach())
    img_ycc_grad = torch.zeros_like(img_ycc)

    for k in range(iterMax):
        img_rgb_k = ycbcr_to_rgb(img_ycc)
        img_rgb_k = torch.clamp(img_rgb_k, min=min_val, max=max_val)
        output = model(img_rgb_k) 
        loss = criterion(output, label) 
        loss.backward()   
        normalized_grad = img_ycc.grad.data / torch.sum(torch.abs(img_ycc.grad.data))
        g_new = mu * g_old + normalized_grad    
        for chan in range(num_chan):
          img_ycc.data[0,chan,...] = img_ycc.data[0,chan,...] + (eps[chan] / iterMax) * g_new[0,chan,...].sign()
          img_ycc.data[0,chan,...] = img_ycc_org[0,chan,...] + \
                                     torch.clamp(img_ycc.data[0,chan,...] - \
                                     img_ycc_org[0,chan,...], min=-eps[chan], max=eps[chan])               
        
        g_old = g_new
        img_ycc.grad.data = torch.zeros_like(img_ycc)
        
    img_rgb = ycbcr_to_rgb(img_ycc)
    img_rgb = torch.clamp(img_rgb, min=min_val, max=max_val)
    return img_rgb



# perform imperceptible attack
max_val = 1.
min_val = 0.
num_chan = 3
epsilon = [args.max_epsilon1/ 255., args.max_epsilon2/ 255., args.max_epsilon3 / 255. ]
criterion = nn.CrossEntropyLoss(reduction="sum").to(device)

print('\n============================================================================')
print(args)
print('\n Attacking face images from directory: %s'%(args.input_dir))
print('============================================================================\n')

for img_idx, img_path in enumerate(img_lists):
    img_ycc = read_image_ycbcr(img_path)   
      
    img_ycc = ycbcr_to_tensor(img_ycc)
    img_ycc = img_ycc.to(device)
    img_ycc.requires_grad = True
        
    img_name = get_file_name(img_path, img_type=args.image_format)
    img_label = img_label_tensor.to(device) 
    print('attacking image: %s' %(img_name+'.png'))     
    img_adv = attack(img_ycc, img_label, model, epsilon)  
    save_image(img_adv, img_name, output_dir)
    
print('\n============================================================================')
print('attacked face images have been saved in %s \n'%(output_dir))

        
     