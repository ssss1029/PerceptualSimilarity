from __future__ import absolute_import

import sys
import scipy
import scipy.misc
import numpy as np
import torch
from torch.autograd import Variable
import models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from util import util
from models import dist_model as dm
import pdb
from scipy.ndimage.interpolation import rotate, shift, zoom
import random
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", type=str, default="imgs_saved")
parser.add_argument("--iterations", type=int, default=400)
parser.add_argument("--max-jitter-shift", type=int, default=7)
parser.add_argument("--max-jitter-color", type=float, default=0.1)
parser.add_argument("--epsilon", type=float, default=0.0)
parser.add_argument("--lamb", type=float, default=0.2)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

# model_path = 'checkpoints/lpips/latest_net_.pth'
# model_path = 'checkpoints/adv_lpips/latest_net_.pth'
model_path = 'checkpoints/adv_lpips_20iterations_traintrunk_lambda0.2/latest_net_.pth'

img_a_path  = 'house.png'
img_b_path  = 'nature.png'

IMG_SIZE = 320

# img_a = util.im2tensor(util.load_image(img_a_path)).cuda() # RGB image from [-1,1]
# img_b = util.im2tensor(util.load_image(img_b_path)).cuda()


def rand_shift_x(img, max_shift):
    shift_x = random.randint(-max_shift, max_shift)

    out = torch.ones_like(img)

    # Handle x-shift
    if shift_x >= 0:
        out[:,:, shift_x:] = img[:,:, :IMG_SIZE - shift_x]
        out[:,:, :shift_x] = 0
    else:
        out[:,:, :shift_x] = img[:,:, -shift_x:]
        out[:,:, shift_x:] = 0
    
    return out

def rand_shift_y(img, max_shift):
    shift_y = random.randint(-max_shift, max_shift)

    out = torch.ones_like(img)

    # Handle y-shift
    if shift_y >= 0:
        out[:,:,:, shift_y:] = img[:,:,:, :IMG_SIZE - shift_y]
        out[:,:,:, :shift_y] = 0
    else:
        out[:,:,:, :shift_y] = img[:,:, :, -shift_y:]
        out[:,:,:, shift_y:] = 0
    
    return out

def rand_shift(img, max_shift):
    x_shifted = rand_shift_x(img, max_shift)
    both_shifted = rand_shift_y(x_shifted, max_shift)
    return both_shifted

def rand_color_shift(img, max_shift):
    shift_c1, shift_c2, shift_c3 = [random.uniform(-max_shift, max_shift) for _ in range(3)]

    img[:, 0, :, :] += shift_c1
    img[:, 1, :, :] += shift_c2
    img[:, 2, :, :] += shift_c3

    return img

def rand_jitter(temp):
    temp = rand_shift(img=temp, max_shift=args.max_jitter_shift)
    temp = rand_color_shift(img=temp, max_shift=args.max_jitter_color)
    return temp

transform_list = [
    transforms.Resize(IMG_SIZE, interpolation=1),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
]
transform = transforms.Compose(transform_list)

img_a = Image.open(img_a_path).convert('RGB')
img_a = transform(img_a).unsqueeze(0).cuda()

img_b = Image.open(img_b_path).convert('RGB')
img_b = transform(img_b).unsqueeze(0).cuda()

img_x = img_a.clone().requires_grad_()

EPSILON = args.epsilon
LAMBDA  = args.lamb
print("EPSILON =", EPSILON, "LAMBDA =", LAMBDA)

print(img_a.shape)
print(img_b.shape)

num_iterations = args.iterations
model = dm.DistModel()
model.initialize(
    model='net-lin', 
    net='vgg', 
    colorspace='RGB', 
	model_path=model_path, 
    use_gpu=True, 
    pnet_rand=False, 
    pnet_tune=False,
	version='0.1', 
    gpu_ids=[0]
)

optimizer = torch.optim.Adam([img_x], lr=1e-3)

plt.ion()


def main():
    for i in range(num_iterations):
        if i == 0:
            pred_img = img_x[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            plt.imsave('{0}/start_image.jpg'.format(args.save_dir), pred_img)

            pred_img = img_a[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            plt.imsave('{0}/a_image.jpg'.format(args.save_dir), pred_img)

            pred_img = img_b[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            plt.imsave('{0}/b_image.jpg'.format(args.save_dir), pred_img)

        jittered_image = rand_jitter(img_x)

        P = 2
        L2_dist_from_b = torch.norm((img_x - img_b), p=P)
        percept_dist_from_a = model.forward(jittered_image, img_a)

        loss_1 = L2_dist_from_b
        loss_2 = LAMBDA * (torch.abs(percept_dist_from_a - EPSILON))
        loss = loss_1 + loss_2

        print(
            "Iteration", i,
            "Current ||x - b||_p^p =", loss_1.item(), 
            "Current | d(x, a) - epsilon | =", loss_2.item()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        if i == num_iterations - 1:
            pred_img = img_x[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            plt.imsave('{0}/final_image.jpg'.format(args.save_dir), pred_img)
        elif i % 100 == 0 or (i < 100 and i % 20 == 0):
            pred_img = img_x[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            plt.imsave('{0}/iteration_{1}.jpg'.format(args.save_dir, i), pred_img)



main()