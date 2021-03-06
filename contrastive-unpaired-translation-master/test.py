





"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import torch
import util.patchify as patchify
import time
from pytorch_lightning import metrics
import pytorch_fid
import matplotlib.pyplot as plt
import scipy.ndimage
import nibabel


import numpy as np

import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # prepare metrics
    fake_key = 'fake_' + opt.direction[-1]
    real_key = 'real_' + opt.direction[-1]

    metricMAE = metrics.MeanAbsoluteError().to(torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu'))
    metricMSE = metrics.MeanSquaredError().to(torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu'))
    for i, data in enumerate(dataset):


        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            # model.parallelize()
            if opt.eval:
                model.eval()
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
            # break


        real_A = data['A']
        real_B = data['B']
        print(data.keys())

        print('Input min', np.amin(real_A.numpy()))
        print('Input max', np.amax(real_A.numpy()))
        print('Input max', real_A.numpy().shape)

        patches = patchify.patchify(real_A.numpy(),1, 512)
        for p in range(len(patches)):

            patch = patches[p]

            # transposed = np.transpose(patch.patch, axes=(0,1,3,2))
            data['A'] = torch.tensor(patch.patch).type(torch.FloatTensor)
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            fake_B = model.fake_B.clone().detach()
            patch.patch = fake_B.cpu().numpy() # get image results


        img_path = model.get_image_paths()     # get image paths

        prediction = patchify.unpatchify(patches, 0, 512)

        new_CT = 4095*prediction - 1024
        new_CT = np.array(new_CT, dtype=np.int)
        ni_image = nibabel.Nifti1Image(new_CT[0,0,:,:], np.eye(4))
        os.makedirs('content/results', exist_ok=True)
        nibabel.save(ni_image, 'content/results/'+img_path[0].split('/')[-1])
        prediction = (prediction-0.5)/0.5
        print(prediction)
        # prediction = np.array(scipy.ndimage.zoom(prediction, (1, 1, 2, 2), order=1),dtype=np.float)
        print('Input min', np.amin(prediction))
        print('Input max', np.amax(prediction))
        print(prediction.shape)

        real_A = (real_A-0.5)/0.5
        real_B = (real_B-0.5)/0.5

        # real_A = np.array(scipy.ndimage.zoom(real_A, (1, 1, 2, 2), order=1), dtype=np.float)
        # real_B = np.array(scipy.ndimage.zoom(real_B, (1, 1, 2, 2), order=1),dtype=np.float)
        visuals = {'real_A': real_A, 'fake_B': torch.tensor(prediction), 'real_B': real_B}


        # apply metrics
        # metricMAE(visuals[fake_key], visuals[real_key])
        # metricMSE(visuals[fake_key], visuals[real_key])

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()  # save the HTML

    # compute metrics

    # mae = metricMAE.compute()
    # mse = metricMSE.compute()

    # print('MAE: ', mae)
    # print('MSE: ', mse)

    fid_paths =  [os.path.join(web_dir, 'images', fake_key), os.path.join(web_dir, 'images', real_key)]
    # fid_value = fid_score.calculate_fid_given_paths(fid_paths,
    #                                                 batch_size=50,
    #                                                 device=None,
    #                                                 dims=2048)
    # print('FID: ', fid_value)


