
from __future__ import absolute_import

import sys
import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from scipy.ndimage import zoom
import fractions
import functools
import skimage.transform
from tqdm import tqdm

from torchvision.utils import save_image

from IPython import embed

from . import networks_basic as networks
import models as util

class DistModel(BaseModel):
    def name(self):
        return self.model_name

    def initialize(self, model='net-lin', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None,
            use_gpu=True, printNet=False, spatial=False, 
            is_train=False, lr=.0001, beta1=0.5, version='0.1', gpu_ids=[0], 
            adversarially_train=False, num_adv_iterations=201, adv_epsilon=10):
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            spatial_shape - if given, output spatial shape. if None then spatial shape is determined automatically via spatial_factor (see below).
            spatial_factor - if given, specifies upsampling factor relative to the largest spatial extent of a convolutional layer. if None then resized to size of input images.
            spatial_order - spline order of filter for upsampling in spatial mode, by default 1 (bilinear).
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        BaseModel.initialize(self, use_gpu=use_gpu, gpu_ids=gpu_ids)

        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model_name = '%s [%s]'%(model,net)

        self.adv_train = adversarially_train
        self.num_adv_iterations = num_adv_iterations
        self.adv_epsilon = adv_epsilon

        self.adversary = PGD_l2(
            epsilon=adv_epsilon,
            num_steps=num_adv_iterations,
            step_size=1e-2
        )

        if(self.model == 'net-lin'): # pretrained net + linear layer
            self.net = networks.PNetLin(pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net,
                use_dropout=True, spatial=spatial, version=version, lpips=True)
            kw = {}
            if not use_gpu:
                kw['map_location'] = 'cpu'
            if(model_path is None):
                import inspect
                model_path = os.path.abspath(os.path.join(inspect.getfile(self.initialize), '..', 'weights/v%s/%s.pth'%(version,net)))

            if(not is_train):
                print('Loading model from: %s'%model_path)
                self.net.load_state_dict(torch.load(model_path, **kw), strict=False)

        elif(self.model=='net'): # pretrained network
            self.net = networks.PNetLin(pnet_rand=pnet_rand, pnet_type=net, lpips=False)
        elif(self.model in ['L2','l2']):
            self.net = networks.L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = networks.DSSIM(use_gpu=use_gpu,colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train: # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = networks.BCERankingLoss()
            self.parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
        else: # test mode
            self.net.eval()

        if(use_gpu):
            self.net.to(gpu_ids[0])
            self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if(self.is_train):
                self.rankLoss = self.rankLoss.to(device=gpu_ids[0]) # just put this on GPU0

        if(printNet):
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''

        return self.net.forward(in0, in1, retPerLayer=retPerLayer)

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if(hasattr(module, 'weight') and module.kernel_size==(1,1)):
                module.weight.data = torch.clamp(module.weight.data,min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

        if(self.use_gpu):
            self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
            self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
            self.input_p1 = self.input_p1.to(device=self.gpu_ids[0])
            self.input_judge = self.input_judge.to(device=self.gpu_ids[0])

        self.var_ref = Variable(self.input_ref,requires_grad=True)
        self.var_p0 = Variable(self.input_p0,requires_grad=True)
        self.var_p1 = Variable(self.input_p1,requires_grad=True)

    def generate_adv_example(self, image):
        image_copy = image.clone()
        img_a = Variable(
            image + ((torch.rand_like(image) - 0.5) * 1e-4),
            requires_grad=False
        )
        img_x = Variable(image, requires_grad=True)
        # save_image(img_x, "/home/saurav/PerceptualSimilarity/img_x_{0}.png".format(0))
        print("Initial ||img_x - img_a||^2  = ", torch.sum((img_a - img_x) ** 2).item())
        adv_optimizer = torch.optim.Adam([img_x], lr=1e-2)
        for i in range(self.num_adv_iterations):
            dist = self.forward(img_x, img_a)
            dist = -1.0 * torch.mean(dist)
            # if i % 100 == 0 and i != 0:
            #     save_image(img_x, "/home/saurav/PerceptualSimilarity/img_x_{0}.png".format(i))
            #     print(dist)
            adv_optimizer.zero_grad()
            dist.backward()
            adv_optimizer.step()
            img_x.data = tensor_clamp_l2(img_x.data, img_a.data, self.adv_epsilon)
        print(
            "||img_x - img_a||^2 = ", torch.sum((img_a - img_x) ** 2).item(), 
            "d(img_x, img_a) = ", torch.mean(self.forward(img_x, image_copy)).item()
        )
        return img_x.data

    # def generate_adv_example(self, image):
    #     img_a = Variable(
    #         image + ((torch.rand_like(image) - 0.5) * 1e-4),
    #         requires_grad=False
    #     )
    #     img_x = Variable(image, requires_grad=True)
    #     save_image(img_x, "/home/saurav/PerceptualSimilarity/img_x_{0}.png".format(0))
    #     print("Initial difference between img_a and img_x = ", torch.sum((img_a - img_x) ** 2))
    #     adv_optimizer = torch.optim.Adam([img_x], lr=1e-2)
    #     for i in range(self.num_adv_iterations):
    #         dist = self.forward(img_x, img_a)
    #         dist = -1.0 * torch.mean(dist)
    #         if i % 100 == 0 and i != 0:
    #             save_image(img_x, "/home/saurav/PerceptualSimilarity/img_x_{0}.png".format(i))
    #             print(dist)
    #         adv_optimizer.zero_grad()
    #         dist.backward()
    #         adv_optimizer.step()
    #         img_x.data = tensor_clamp_l2(img_x.data, img_a.data, self.adv_epsilon)
    #         return img_x.data

    def forward_train(self): # run forward pass
        # print(self.net.module.scaling_layer.shift)
        # print(torch.norm(self.net.module.net.slice1[0].weight).item(), torch.norm(self.net.module.lin0.model[1].weight).item())

        self.d0 = self.forward(self.var_ref, self.var_p0)
        self.d1 = self.forward(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0,self.d1,self.input_judge)

        self.var_judge = Variable(1.*self.input_judge).view(self.d0.size())

        self.loss_total = self.rankLoss.forward(self.d0, self.d1, self.var_judge*2.-1.)

        if self.adv_train:
            # print("Generating adv example")
            # ref_adversarial_example = self.adversary(
            #     self.input_ref,
            #     loss_function = lambda x: torch.mean(self.forward(x, self.var_ref))
            # )
            # dx = torch.mean(self.forward(
            #     self.var_ref,
            #     Variable(ref_adversarial_example, requires_grad=True)
            # ))
            # print(dx)
            # print(self.loss_total)
            # exit()

            print("Generating adv example")
            ref_adversarial_example = self.generate_adv_example(self.input_ref.clone())
            adv_loss = self.forward(
                self.input_ref, 
                ref_adversarial_example
            )
            self.loss_total += torch.mean(adv_loss)

        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self,d0,d1,judge):
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1<d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0*judge_per + (1-d1_lt_d0)*(1-judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()),
                            ('acc_r', self.acc_r)])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256/self.var_ref.data.size()[2]

        ref_img = util.tensor2im(self.var_ref.data)
        p0_img = util.tensor2im(self.var_p0.data)
        p1_img = util.tensor2im(self.var_p1.data)

        ref_img_vis = zoom(ref_img,[zoom_factor, zoom_factor, 1],order=0)
        p0_img_vis = zoom(p0_img,[zoom_factor, zoom_factor, 1],order=0)
        p1_img_vis = zoom(p1_img,[zoom_factor, zoom_factor, 1],order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis),
                            ('p1', p1_img_vis)])

    def save(self, path, label):
        if(self.use_gpu):
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    def update_learning_rate(self,nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type,self.old_lr, lr))
        self.old_lr = lr

def score_2afc_dataset(data_loader, func, name=''):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches 
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    '''

    d0s = []
    d1s = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        d0s+=func(data['ref'],data['p0']).data.cpu().numpy().flatten().tolist()
        d1s+=func(data['ref'],data['p1']).data.cpu().numpy().flatten().tolist()
        gts+=data['judge'].cpu().numpy().flatten().tolist()

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    return(np.mean(scores), dict(d0s=d0s,d1s=d1s,gts=gts,scores=scores))

def score_jnd_dataset(data_loader, func, name=''):
    ''' Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return pytorch array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    '''

    ds = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        ds+=func(data['p0'],data['p1']).data.cpu().numpy().tolist()
        gts+=data['same'].cpu().numpy().flatten().tolist()

    sames = np.array(gts)
    ds = np.array(ds)

    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]

    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1-sames_sorted)
    FNs = np.sum(sames_sorted)-TPs

    precs = TPs/(TPs+FPs)
    recs = TPs/(TPs+FNs)
    score = util.voc_ap(recs,precs)

    return(score, dict(ds=ds,sames=sames))


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

def normalize_l2(x):
    """
    Expects x.shape == [N, C, H, W]
    """
    norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
    norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return x / norm

class PGD_l2(nn.Module):
    def __init__(self, epsilon, num_steps, step_size):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def forward(self, bx, loss_function):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        init_noise = normalize_l2(torch.randn(bx.size()).cuda()) * (np.random.rand() - 0.5) * self.epsilon
        adv_bx = (bx + init_noise).clamp(-1, 1).requires_grad_()

        for i in range(self.num_steps):
            loss = loss_function(adv_bx)
            if i % 100 == 0:
                print("Iteration", i, "dist (aka loss) = ", loss)
            grad = normalize_l2(torch.autograd.grad(loss, adv_bx, only_inputs=True)[0])
            adv_bx = adv_bx + self.step_size * grad
            adv_bx = tensor_clamp_l2(adv_bx, bx, self.epsilon).clamp(-1, 1)
            adv_bx = adv_bx.data.requires_grad_()

        return adv_bx.detach()
