"""
This script is used to repurpose pretrained resnet9 generators of cycleGAN
and implant them in an already initializes resnet9 generator for the CUT model.

Author: David Helm
Date 28/12/2020
"""

import torch
import os

import argparse

parser = argparse.ArgumentParser(description='Conversion Script')
parser.add_argument('--cycleGAN_path',  type=str,help='path of the cycleGAN model')
parser.add_argument('--CUT_path',  type=str,help='path of the cycleGAN model')
parser.add_argument('--output_path',  type=str, help='path of the cycleGAN model')

args = parser.parse_args()

def implant_weights(model_cycleGAN_path, model_CUT_path, output_path):

    model_cG = torch.load(model_cycleGAN_path)
    model_CUT = torch.load(model_CUT_path)
    # TODO: do initialization here
    model_adapted = model_CUT.copy()
    for key, value in model_cG.items():
    
        try:
            if len(model_CUT['module.'+key]) == len(model_cG[key]):
                print('module.'+key)
                model_adapted[key] = model_cG[key]
                del model_adapted['module.'+ key]
            else: print(key + ' hes a different length')
        except: 
            KeyError
            print(key + ' does not exist')

    print(len(model_adapted.keys()))
    keys = list(model_adapted.keys())

    for key in keys:
        if key[:7] == 'module.':
            model_adapted[key[7:]] = model_adapted[key]
            del model_adapted[key]
            print(key)

    print(len(model_adapted.keys()))

    # some manual values, this should be investigated such that it is no longer necessary

    model_adapted['model.1.weight'] = model_CUT['module.model.1.weight']
    model_adapted['model.22.weight'] = model_CUT['module.model.22.weight']
    model_adapted['model.30.weight'] = model_CUT['module.model.30.weight']

    # TODO: make some checks
    os.mkdir(output_path)

    torch.save(model_adapted, output_path+'/latest_net_G.pth', _use_new_zipfile_serialization=False)



if __name__ == "__main__":
    print(args)
    implant_weights(args.cycleGAN_path, args.CUT_path, args.output_path)