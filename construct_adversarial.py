from __future__ import absolute_import

import sys
import scipy
import scipy.misc
import numpy as np
import torch
from torch.autograd import Variable
import models
import matplotlib.pyplot as plt


"""
Given one image - make two copies (img_a and img_x)
Keep img_a fixed and max d(img_a, img_x) subject to ||a - x||^2 < EPSILON
"""

use_gpu = True

img_path  = './imgs/ex_ref.png'

img_x = scipy.misc.imread(img_path).transpose(2, 0, 1) / 255.
img_a = scipy.misc.imread(img_path).transpose(2, 0, 1) / 255.

# Torchify
EPSILON = 5.0
print(EPSILON)

img_a = Variable(torch.FloatTensor(img_a)[None,:,:,:], requires_grad=False)
img_x = Variable(
    torch.FloatTensor(img_x)[None,:,:,:] + \
    (torch.rand_like(torch.FloatTensor(img_x)[None,:,:,:]) - 0.5) * 1e-5, 
    requires_grad=True
)

loss_fn = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=use_gpu)
optimizer = torch.optim.SGD([img_x], lr=1e-2, momentum=0.9, nesterov=True)

plt.ion()
# fig = plt.figure(1)
# ax = fig.add_subplot(131)
# ax.imshow(ref_img.transpose(1, 2, 0))
# ax.set_title('target')
# ax = fig.add_subplot(133)
# ax.imshow(pred_img.transpose(1, 2, 0))
# ax.set_title('initialization')

def main():
    for i in range(10000):
        dist = -1 * loss_fn.forward(img_x, img_a, normalize=True)
        optimizer.zero_grad()
        dist.backward()
        optimizer.step()
        img_x.data = tensor_clamp_l2(img_x.data, img_a.data, EPSILON)
        
        if i % 100 == 0:
            print('iter %d, dist %.3g' % (i, dist.view(-1).data.cpu().numpy()[0]))
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