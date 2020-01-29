import argparse
import os
import models
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image', type=str)
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='vgg',use_gpu=opt.use_gpu)

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir1)

dists = []

for file in files:
	# Load images
	img0 = util.im2tensor(util.load_image(os.path.join(opt.image))) # RGB image from [-1,1]
	img1 = util.im2tensor(util.load_image(os.path.join(opt.dir1,file)))

	if(opt.use_gpu):
		img0 = img0.cuda()
		img1 = img1.cuda()

	# Compute distance
	dist01 = model.forward(img0,img1)
	dists.append(dist01.cpu().detach().item())
	print('%s: %.3f'%(file,dist01))
	f.writelines('%s: %.6f\n'%(file,dist01))

print("Average = ", sum(dists) / float(len(dists)))

f.close()
