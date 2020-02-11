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
from models import dist_model as dm
from scipy.ndimage.interpolation import rotate, shift, zoom
import random
import os

"""
Given one image - make two copies (img_a and img_x)
Keep img_a fixed and max d(img_a, img_x) subject to ||a - x||^2 < EPSILON
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", type=str, default="imgs_saved")
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--iterations", type=int, default=400)
parser.add_argument("--max-jitter-shift", type=int, default=7)
parser.add_argument("--max-jitter-color", type=float, default=0.1)
parser.add_argument("--epsilon", type=float, default=0.2)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)


use_gpu = True

img_path  = 'house.png'
# model_path = 'checkpoints/lpips/latest_net_.pth'
model_path = 'checkpoints/adv_lpips/latest_net_.pth'
# model_path = 'checkpoints/adv_lpips_20iterations_traintrunk_lambda0.2/latest_net_.pth'
IMG_SIZE = 320

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

img_x = Image.open(img_path).convert('RGB')
print(img_x.size)
img_x = transform(img_x)

img_a = Image.open(img_path).convert('RGB')
img_a = transform(img_a)

EPSILON = args.epsilon
print(EPSILON)

print(img_x.shape)
print(img_a.shape)

img_a = Variable(
    torch.FloatTensor(img_a)[None,:,:,:] + \
    (torch.rand_like(torch.FloatTensor(img_x)[None,:,:,:]) - 0.5) * 1e-6, 
    requires_grad=False
)
img_x = Variable(torch.FloatTensor(img_x)[None,:,:,:], requires_grad=True)
print(img_x.shape)
print(img_a.shape)



print("Initial difference between img_a and img_x = ", torch.sum((img_a - img_x) ** 2))

num_iterations = args.iterations

# initialize model
model = dm.DistModel()
# model.initialize(model=opt.model,net=opt.net,colorspace=opt.colorspace,model_path=opt.model_path,use_gpu=opt.use_gpu)
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

# optimizer = torch.optim.SGD([img_x], lr=1e-2, momentum=0.9, nesterov=False)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer,
#     [5000],
#     gamma=0.1
# )

plt.ion()

def main():
    global img_x
    batch_size = args.batch_size
    optimizer = torch.optim.Adam([img_x], lr=1e-2)

    for i in range(num_iterations):        

        if i == 0:
            pred_img = img_x[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            # ax = fig.add_subplot(132)            
            # ax.imshow(pred_img)
            # ax.set_title('iter %d, dist %.3f' % (i, dist.view(-1).data.cpu().numpy()[0]))
            # plt.pause(5e-2)
            plt.imsave('%s/A2_attck_iter_%04d.jpg' % (args.save_dir, i), pred_img)
        
        jittered_images = []
        base_images = []
        for _ in range(batch_size):
            jittered_images.append(rand_jitter(img_x))
            base_images.append(img_a.clone())
        
        jittered_images = torch.cat(jittered_images, 0)
        base_images = torch.cat(base_images, 0)

        dist = -1.0 * torch.sum(model.forward(jittered_images, base_images)) / batch_size
        optimizer.zero_grad()
        dist.backward()
        optimizer.step()
        # scheduler.step()
        img_x.data = tensor_clamp_l2(img_x.data, img_a.data, EPSILON)
        
        if True:
            print('iter %d, dist %f' % (i, dist.view(-1).data.cpu().numpy()[0]))
            pred_img = img_x[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            # ax = fig.add_subplot(132)            
            # ax.imshow(pred_img)
            # ax.set_title('iter %d, dist %.3f' % (i, dist.view(-1).data.cpu().numpy()[0]))
            # plt.pause(5e-2)
            if i == num_iterations - 1 or i % 100 == 0 or (i < 100 and i % 20 == 0):
                plt.imsave('%s/A2_attck_iter_%04d.jpg' % (args.save_dir, i),pred_img)


def tensor_clamp_l2(x, center, radius):
    """batched clamp of x into l2 ball around center of given radius"""
    x = x.data
    diff = x - center
    diff_norm = torch.norm(diff.view(diff.size(0), -1), p=2, dim=1)
    project_select = diff_norm > radius
    if project_select.any():
        diff_norm = diff_norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        new_x = x
        new_x[project_select] = (center + (diff / diff_norm) * radius)[project_select]
        return new_x
    else:
        return x

def tensor_clamp_l_infinity(x, center, radius):
    return torch.min(torch.max(x, center - radius), center + radius).clamp(0, 1)

main()