
from __future__ import print_function

import copy
import os.path as osp

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.hub
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import sys
sys.path.append('../../')
import models as models
import argparse
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-i', '--image', default='cat1.jpg', type=str)
parser.add_argument('-l', '--layer', default='layer3.5.se', type=str)
parser.add_argument('-t', '--target', default='se.conv', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='se_resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: se_resnet50)')
parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')


args = parser.parse_args()
def main():
    global args

    distributed_model = False
    pytorch_1_2 = False

    if args.checkpoint is None:
        args.checkpoint = '/home/{PATH}/checkpoints/imagenet/' + args.arch+'/model_best.pth.tar'

    t = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                            ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    model = models.__dict__[args.arch]()
    model = model.cuda()

    check_point = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(0))
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_check_point = OrderedDict()
    for k, v in check_point['state_dict'].items():
        # name = k[7:]  # remove `module.`
        # name = k[9:]  # remove `module.1.`
        # if pytorch_1_2:
        #     name = k[7:]  # remove `module.`
        # new_check_point[name] = v

        if args.target in k:
            # # dca_se_resnet50 se.conv.0.weight
            # # print("{}-{}-{}".format(k,v[0,0,0,0],v[0,1,0,0]))
            # # print(v.size())
            temp = torch.abs(v[0,1,0,0])/(torch.abs(v[0,1,0,0])+torch.abs(v[0,0,0,0]))
            print(temp.cpu().numpy())

if __name__ == '__main__':
    main()
