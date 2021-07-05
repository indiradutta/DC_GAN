import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import os
import json
import gdown

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from DC_GAN.dcgan.dcgan import Generator

__PREFIX__ = os.path.dirname(os.path.realpath(__file__))

class Deep_Conv_GAN(object):

    def __init__(self, nc = 3, nz = 100, ngf = 64, ngpu = 1):

        ''' 
        The constructor has the Parameters which are going to be used to generate the images

        Parameters:

        - nc(default: 3): number of color channels in an image, we have used 3 channels(RGB).

        - nz(default: 100): length of the latent vector that is initially passed into the Generator, according to the paper it is 100.

        - ngf(default: 64):  denotes the depth of the feature maps passed through the Generator, according to the paper it is 64.

        - ndf(default: 64): denotes the depth of the feature maps passed through the Discriminator, according to the paper it is 64.

        - ngpu(default: 1): number of GPUs available for training. If no GPU is available, the model will train on CPU. Here, we have only 1 GPU available.
        '''

        if ngpu > 0 and not torch.cuda.is_available():
            raise ValueError('ngpu > 0 but cuda not available')

        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.ngpu = ngpu
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    def inference(self, set_weight_dir = 'dcgan-model.pth', set_gen_dir = 'result_img'):

        set_weight_dir = __PREFIX__ + "/weights/" + set_weight_dir

        ''' saving generated images in a directory '''
        def save_image(set_gen_dir):
            if os.path.exists(set_gen_dir):
                print("Found directory for saving generated images")
                return 1
            else:
                print("Directory for saving images not found, making a directory named 'result_img'")
                os.mkdir(set_gen_dir)
                return 1
        
        ''' checking if weights are present '''
        def check_weights(set_weight_dir):
            if os.path.exists(set_weight_dir):
                print("Found weights")
                return 1
            else:
                print("Downloading weights")
                download_weights()

        ''' downloading weights if not present '''
        def download_weights():
            with open(__PREFIX__+"/config/weights_download.json") as fp:
                json_file = json.load(fp)
                if not os.path.exists(__PREFIX__+"/weights/"):
                    os.mkdir(__PREFIX__+"/weights/")
                url = 'https://drive.google.com/uc?id={}'.format(json_file['dcgan-model.pth'])
                gdown.download(url, __PREFIX__+"/weights/dcgan-model.pth", quiet=False)
                set_weight_dir = "dcgan-model.pth"
                print("Download finished")

        ''' checking if weights are present '''
        check_weights(set_weight_dir)

        '''saving the generated images '''
        save_image(set_gen_dir)

        '''calling the DCGAN for inference '''
        model_GAN = Generator(1, 100, 64, 3)

        ''' uploading the model '''
        checkpoint = torch.load(set_weight_dir)
        model_GAN.load_state_dict(checkpoint['generator_state_dict'])
        model_GAN.eval()

        ''' saving the generated images'''
        def save_new_img():

            b_size = 512
            noise = torch.randn(b_size, 100, 1, 1)
            out = model_GAN(noise).detach().cpu()
            print("The generated images are saved in the given directory")

            ''' saving the generated images in a list '''
            img_list = []
            for i in range(b_size):
                img_list.append(out[i,:,:,:])
            
            ''' saving the generated images in jpg format '''
            for i in range(len(img_list)):
                generated_image = '{}/generated_image_{}.jpg'.format(set_gen_dir,i)
                vutils.save_image(img_list[i], generated_image, padding = 0)            

        save_new_img()
