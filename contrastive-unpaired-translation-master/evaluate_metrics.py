import numpy as np
import cv2
import nibabel as nib
import os
import yaml
from pytorch_fid import fid_score

#from pytorch_lightning import metrics

import argparse

parser = argparse.ArgumentParser(description='Metrics Script')
parser.add_argument('--bodymask_path',  type=str)
parser.add_argument('--real_slices_path',  type=str)
parser.add_argument('--fake_slices_path',  type=str)
parser.add_argument('--results_path',  type=str, default='./metrics_result.yaml')
parser.add_argument('--FID', action='store_true')
parser.add_argument('--debug_images', action='store_true')

args = parser.parse_args()

def build_volume(slices, patient):
    # build 3D volume w*h*d
    names_slices = slices.keys()
    filtered_slice_keys =[s for s in names_slices if (s[:4] == patient)]

    filtered_slice_keys.sort()
    volume_dimensions = (slices[filtered_slice_keys[0]].shape[0], slices[filtered_slice_keys[0]].shape[0], len(filtered_slice_keys))

    volume = np.zeros(volume_dimensions)

    for i, sl in enumerate(filtered_slice_keys):
        if len(slices[sl].shape) == 2:
            volume[:,:,i] = slices[sl]
        else:
            volume[:,:,i] = slices[sl][:,:,0]

    return volume


def read_slices(path):
    # assume folder is for one patient
    filelist = os.listdir(path)
    imglist = {}
    for f in filelist:
        if f[-4:] == '.png':
            imglist[f] = cv2.imread(os.path.join(path, f))
            
        elif f[-4:] == '.nii':
            A_img_nifti = nib.load(os.path.join(path, f))
            imglist[f] = A_img_nifti.get_fdata(caching = "unchanged").astype(np.int16)

            #print(imglist[f][0,0])
        else:
            print(f + ' does not fit specified input')
    return imglist


def rescale_slices(slices):
    for key in slices:
        slices[key] = 4095*slices[key] -1024
        
    return slices


def read_mask(path):
    # assume folder is for one patient
    filelist = os.listdir(path)
    imglist = {}
    
    for f in filelist:
        A_img_nifti = nib.load(os.path.join(path, f))
        imglist[f] = A_img_nifti.get_fdata(caching = "unchanged")
    return imglist


def mask_volume(volume, mask):
    return np.multiply(volume, mask)


if __name__ == "__main__":
    bodymask_path = args.bodymask_path
    real_slices_path = args.real_slices_path
    fake_slices_path = args.fake_slices_path
    results_path = args.results_path
    print('reading real slices from: '+ real_slices_path)
    real_slices = read_slices(real_slices_path)
    print('reading fake slices from: '+ fake_slices_path)
    fake_slices = read_slices(fake_slices_path)
    print('rescaling fake slices')

    mask_slices = read_slices(bodymask_path)
    results = {}
    results['real_path'] = real_slices_path
    results['fake_path'] = fake_slices_path
    results['masks_path'] = bodymask_path

    # patient list with registered data, this needs to be changed if more
    # or other registered training data is available. The all the files with the
    # filename beginning with one of the list elements are considered.

    pat = ['PAT1', 'PAT3']

    diff_all_pat = []
    for p in pat:
        # make volumes
        print(p)
        real_vol = build_volume(real_slices, p)        
        fake_vol = build_volume(fake_slices, p)

        mask = build_volume(mask_slices, p)

        # mask volumes
        fake_vol = mask_volume(fake_vol, mask)
        real_vol = mask_volume(real_vol, mask)

        diff = real_vol - fake_vol
        diff_masked = diff[mask>0]
        diff_all_pat.append(diff_masked)
        # calculate MAE
        mae = (np.abs(diff_masked)).mean()
        sd_ae = np.std(np.abs(diff_masked))

        print(p + ' MAE: '+ str(mae) + ', SD: '+ str(sd_ae))

        # calculate MSE
        mse = ((diff_masked)**2).mean()
        sd_se = np.std((diff_masked)**2)
        print(p + ' MSE: '+ str(mse) + ', SD: '+ str(sd_se))

        # calculate ME
        me = diff_masked.mean()
        sd_e = np.std(diff_masked)
        print(p + ' ME: '+ str(me)+ ', SD: '+ str(sd_e))

        # calculate MRSE
        mrse = np.sqrt(mse)
        print(p + ' MRSE: '+ str(mrse))
        
        # add to dict
        results[p] = {
            'mse': float(mse), 'sd_mse': float(sd_se), 
            'mae': float(mae), 'sd_mae': float(sd_ae),
            'me': float(me), 'sd_me': float(sd_e)}
    
    # calculations for all patients
    p = 'PAT1+PAT3'
    # calculate MAE 
    diff_all_pat = np.concatenate(diff_all_pat)
    # calculate MAE
    mae = (np.abs(diff_all_pat)).mean()
    sd_ae = np.std(np.abs(diff_all_pat))

    print(p + ' MAE: '+ str(mae) + ', SD: '+ str(sd_ae))

    # calculate MSE
    mse = ((diff_all_pat)**2).mean()
    sd_se = np.std((diff_all_pat)**2)
    print(p + ' MSE: '+ str(mse) + ', SD: '+ str(sd_se))


    # calculate ME
    me = diff_all_pat.mean()
    sd_e = np.std(diff_all_pat)
    print(p + ' ME: '+ str(me)+ ', SD: '+ str(sd_e))
    results[p] = {
        'mse': float(mse), 'sd_mse': float(sd_se), 
        'mae': float(mae), 'sd_mae': float(sd_ae),
        'me': float(me), 'sd_me': float(sd_e)}


    print('FID Calculations: ' + str(args.FID))
    if args.FID:
        print('Calculating FID score, this may take a while...')
        fid_paths =  [real_slices_path,fake_slices_path]
        fid_value = fid_score.calculate_fid_given_paths(fid_paths,
                                                     batch_size=50,
                                                     device=None,
                                                     dims=2048)
        results['FID'] = float(fid_value)
    with open(results_path, 'w') as file:
        documents = yaml.dump(results, file)
    print(results)

    if args.debug_images:
        def on_change(i):   
            img = np.hstack(((real_vol[:,:,i]+1024)/4095, (fake_vol[:,:,i]+1024)/4095, mask[:,:,i]))
            cv2.imshow('real-fake-mask', img)
        img = np.hstack(((real_vol[:,:,0]+1024)/4095, (fake_vol[:,:,0]+1024)/4095, mask[:,:,0]))
        cv2.imshow('real-fake-mask', img)
        cv2.createTrackbar('slider', 'real-fake-mask', 0, 299, on_change)
        while True:
            k = cv2.waitKey(0)
            if k==27:    # Esc key to stop
                cv2.destroyAllWindows()
                break


    

