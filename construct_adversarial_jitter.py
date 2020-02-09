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

"""
Given one image - make two copies (img_a and img_x)
Keep img_a fixed and max d(img_a, img_x) subject to ||a - x||^2 < EPSILON
"""

use_gpu = True

img_path  = './dataset/2afc/val/cnn/ref/000777.png'
# model_path = 'checkpoints/adv_lpips/latest_net_.pth'
model_path = 'checkpoints/adv_lpips_20iterations_traintrunk_lambda0.2/latest_net_.pth'
IMG_SIZE = 256


def rand_jitter(temp):
    temp = shift(temp, shift=(0, random.randint(-10, 10), random.randint(-10, 10)))
    temp = rotate(temp, angle = np.random.randint(-10,10,1), axes=(1, 2), reshape=False)
    return temp

transform_list = [
    transforms.Scale(IMG_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
]
transform = transforms.Compose(transform_list)

img_x = Image.open(img_path).convert('RGB')
img_x = transform(img_x)

img_a = Image.open(img_path).convert('RGB')
img_a = transform(img_a)

img_x = rand_jitter(img_x)
img_a = img_x.copy()

EPSILON = 0.1
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

num_iterations = 400

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
optimizer = torch.optim.Adam([img_x], lr=1e-2)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer,
#     [5000],
#     gamma=0.1
# )

plt.ion()

def main():
    for i in range(num_iterations):

        if i == 0:
            pred_img = img_x[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            # ax = fig.add_subplot(132)            
            # ax.imshow(pred_img)
            # ax.set_title('iter %d, dist %.3f' % (i, dist.view(-1).data.cpu().numpy()[0]))
            # plt.pause(5e-2)
            plt.imsave('imgs_saved/A2_attck_iter_%04d.jpg'%i,pred_img)

        dist = -1 * model.forward(img_x, img_a)
        optimizer.zero_grad()
        dist.backward()
        optimizer.step()
        # scheduler.step()
        img_x.data = tensor_clamp_l_infinity(img_x.data, img_a.data, EPSILON)
        
        if i % 100 == 0 or True:
            print('iter %d, dist %f' % (i, dist.view(-1).data.cpu().numpy()[0]))
            pred_img = img_x[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            # ax = fig.add_subplot(132)            
            # ax.imshow(pred_img)
            # ax.set_title('iter %d, dist %.3f' % (i, dist.view(-1).data.cpu().numpy()[0]))
            # plt.pause(5e-2)
            if i == num_iterations - 1 or i % 100 == 0:
                plt.imsave('imgs_saved/A2_attck_iter_%04d.jpg'%i,pred_img)


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