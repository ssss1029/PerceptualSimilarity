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

model_path = 'checkpoints/adv_lpips_20iterations_traintrunk_lambda0.2/latest_net_.pth'
# model_path = 'checkpoints/lpips/latest_net_.pth'

img_a_path  = './dataset/2afc/val/cnn/ref/000012.png'
img_b_path  = './dataset/2afc/val/cnn/ref/000772.png'

# img_a = util.im2tensor(util.load_image(img_a_path)).cuda() # RGB image from [-1,1]
# img_b = util.im2tensor(util.load_image(img_b_path)).cuda()


def rand_jitter(temp):
    temp = shift(temp, shift=(0, 0, random.randint(-10, 10), random.randint(-10, 10)))
    temp = rotate(temp, angle = np.random.randint(-10,10,1), axes=(2, 3), reshape=False)
    return temp

transform_list = [
    transforms.Scale(256),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
]
transform = transforms.Compose(transform_list)

img_a = Image.open(img_a_path).convert('RGB')
img_a = transform(img_a).unsqueeze(0).cuda()

img_b = Image.open(img_b_path).convert('RGB')
img_b = transform(img_b).unsqueeze(0).cuda()

# img_a = torch.from_numpy(rand_jitter(img_a)).cuda()
# img_b = torch.from_numpy(rand_jitter(img_b)).cuda()

img_x = img_a.clone().requires_grad_()
# img_x = 2.0 * (torch.rand_like(img_b, device="cuda").requires_grad_() - 0.5)

# pdb.set_trace()

EPSILON = 0.0
LAMBDA  = 1.0
print("EPSILON =", EPSILON, "LAMBDA =", LAMBDA)

print(img_a.shape)
print(img_b.shape)

num_iterations = 1000
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
optimizer = torch.optim.Adam([img_x], lr=1e-3)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer,
#     [5000],
#     gamma=0.1
# )

plt.ion()

# print(
#     "d(a, x) = ", model.forward(img_a, img_x).item(),
#     "d(b, x) = ", model.forward(img_b, img_x).item(),
#     "d(a, b) = ", model.forward(img_a, img_b).item()
# )

# print(
#     "d(a + noise, a) = ", model.forward(img_a + torch.rand_like(img_a) * 1e-8, img_a).item()
# )
# exit()

def main():
    for i in range(num_iterations):
        if i == 0:
            pred_img = img_x[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            plt.imsave('imgs_saved_A1_attack/start_image.jpg', pred_img)

            pred_img = img_a[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            plt.imsave('imgs_saved_A1_attack/a_image.jpg', pred_img)

            pred_img = img_b[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            plt.imsave('imgs_saved_A1_attack/b_image.jpg', pred_img)


        # L2_dist_from_b = torch.sum((img_x - img_b) ** 2)
        L2_dist_from_b = torch.norm((img_x - img_b), p=50)
        percept_dist_from_a = model.forward(img_x, img_a)

        loss_1 = L2_dist_from_b
        loss_2 = (torch.abs(percept_dist_from_a - EPSILON))

        print(
            "Iteration", i,
            "Current ||x - b||^2 =", L2_dist_from_b.item(), 
            "Current d(x, a) =", percept_dist_from_a.item()
        )

        loss = loss_1 + (LAMBDA * loss_2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        if i == num_iterations - 1:
            pred_img = img_x[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            plt.imsave('imgs_saved_A1_attack/final_image.jpg', pred_img)
        elif i % 100 == 0 or (i < 100 and i % 20 == 0):
            pred_img = img_x[0].data.cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0, 1)
            plt.imsave('imgs_saved_A1_attack/iteration_{0}.jpg'.format(i), pred_img)



main()