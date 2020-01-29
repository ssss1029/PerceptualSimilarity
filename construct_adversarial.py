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


"""
Given one image - make two copies (img_a and img_x)
Keep img_a fixed and max d(img_a, img_x) subject to ||a - x||^2 < EPSILON
"""

use_gpu = True

img_path  = './dataset/2afc/val/cnn/ref/000002.png'

transform_list = [
    transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
]
transform = transforms.Compose(transform_list)

img_x = Image.open(img_path).convert('RGB')
img_x = transform(img_x)

img_a = Image.open(img_path).convert('RGB')
img_a = transform(img_a)

# Torchify
EPSILON = 5.0
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


num_iterations = 10000
loss_fn = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=use_gpu, version="0.1")
# optimizer = torch.optim.SGD([img_x], lr=1e-2, momentum=0.9, nesterov=False)
optimizer = torch.optim.Adam([img_x], lr=1e-2)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer,
#     [5000],
#     gamma=0.1
# )

plt.ion()
# fig = plt.figure(1)
# ax = fig.add_subplot(131)
# ax.imshow(ref_img.transpose(1, 2, 0))
# ax.set_title('target')
# ax = fig.add_subplot(133)
# ax.imshow(pred_img.transpose(1, 2, 0))
# ax.set_title('initialization')

def main():
    for i in range(num_iterations):
        dist = -1 * loss_fn.forward(img_x, img_a, normalize=True)
        optimizer.zero_grad()
        dist.backward()
        optimizer.step()
        # scheduler.step()
        img_x.data = tensor_clamp_l2(img_x.data, img_a.data, EPSILON)
        
        if i % 100 == 0:
            print('iter %d, dist %f' % (i, dist.view(-1).data.cpu().numpy()[0]))
            pred_img = img_x[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            # ax = fig.add_subplot(132)            
            # ax.imshow(pred_img)
            # ax.set_title('iter %d, dist %.3f' % (i, dist.view(-1).data.cpu().numpy()[0]))
            # plt.pause(5e-2)
            plt.imsave('imgs_saved/%04d.jpg'%i,pred_img)


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


main()