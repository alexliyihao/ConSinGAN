import os
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
from ConSinGAN.config import get_arguments
import ConSinGAN.functions as functions
import ConSinGAN.models as models
from ConSinGAN.imresize import imresize, imresize_to_shape


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def generate_samples(netG, reals_shapes, noise_amp, opt, fixed_noise = 0, scale_w=1.0, scale_h=1.0, reconstruct=False, n=50):
    if reconstruct:
        reconstruction = netG(fixed_noise, reals_shapes, noise_amp)
        return reconstruction
    samples = []
    for idx in range(n):
        noise = functions.sample_random_noise(opt.train_stages - 1, reals_shapes, opt)
        sample = netG(noise, reals_shapes, noise_amp)
        samples.append(functions.convert_image_np(sample.detach()))
    return np.array(samples)


def generate_image(model_dir, num_samples):
    parser = get_arguments()
    parser.add_argument('--model_dir', help='input image name', required=True)
    parser.add_argument('--gpu', type=int, help='which GPU', default=0)
    parser.add_argument('--num_samples', type=int, help='which GPU', default=50)
    parser.add_argument('--naive_img', help='naive input image  (harmonization or editing)', default="")

    opt = parser.parse_args(["--model_dir", model_dir, "--num_samples", str(num_samples)])
    _gpu = opt.gpu
    _naive_img = opt.naive_img
    __model_dir = opt.model_dir
    opt = functions.load_config(opt)
    opt.gpu = _gpu
    opt.naive_img = _naive_img
    opt.model_dir = __model_dir

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        opt.device = "cuda:{}".format(opt.gpu)

    dir2save = os.path.join(opt.model_dir, "Evaluation")
    make_dir(dir2save)

    print("Loading models...")
    netG = torch.load('%s/G.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    fixed_noise = torch.load('%s/fixed_noise.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    reals = torch.load('%s/reals.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    noise_amp = torch.load('%s/noise_amp.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    reals_shapes = [r.shape for r in reals]


    print("Generating Samples...")
    with torch.no_grad():
        # # generate reconstruction
        generate_samples(netG, reals_shapes, noise_amp, opt, fixed_noise = fixed_noise,reconstruct = True)
        # generate random samples of normal resolution
        rs0 = generate_samples(netG, reals_shapes, noise_amp, opt, n=opt.num_samples)

    return rs0
