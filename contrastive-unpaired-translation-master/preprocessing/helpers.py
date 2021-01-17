# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:54:59 2019

@author: lagag
"""

import os 
import pandas as pd
import pydicom
import re
import string
import glob
from pydicom import dcmread
import numpy as np
import scipy
from scipy import ndimage
from skimage.draw import polygon
from skimage.measure import label
from skimage.segmentation import flood, flood_fill
from scipy.ndimage import morphology, measurements, filters
import nibabel as nib
from scipy.ndimage import morphology, measurements, filters, \
    binary_opening, binary_closing, binary_erosion, binary_dilation, binary_fill_holes
import net
from skimage.measure import regionprops, label
import shutil

def get_sequences(path_patient, new_path_patient, files_dcm):
    if not os.path.exists(new_path_patient):
        os.makedirs(new_path_patient)
    names = []
    new_series = []
    count = 0
    for file in files_dcm:
        old_path = os.path.join(path_patient,file)
        print(old_path)
        ds = dcmread(old_path, force = True)

        try:
            name = str(ds[0x0008,0x103E].value)
            print(name)
        except:
            try:
                name = str(ds['SeriesDescription'].value)
            except:
                print("Problem reading dicom in {}".format(path_patient))
                continue
        new_path_patient2 = os.path.join(new_path_patient, name)
        print(new_path_patient2)
        if name not in names:
            names.append(name)
        if not os.path.exists(new_path_patient2):
            os.makedirs(new_path_patient2)
        shutil.copy2(old_path, new_path_patient2)
    return names


def augment_crops(array, affine, new_dir):
        # crop_array_orig[crop_array_orig > 1800] = 0
        crop_array_orig = array.astype(np.int16)
        # crop_new = nib.Nifti1Image(crop_array_orig, affine)
        # nib.save(crop_new, os.path.join(new_dir, "crop_centered_0.9765_2.nii"))

        imageFlipper = sampling.imagetransformer.Flipper((0.5,0.,0.))
        # Data augmentation: Rotations of up to +-10 degrees with a probability of 0.75
        imageRotator = net.sampling.imagetransformer.Rotator((0.75, 0.75, 0.),(15,15,0))
        # imageDeformer = sampling.imagetransformer.ElasticDeformer(1.,2,2)
        imageShifter = sampling.imagetransformer.RandomShifter((1.,1.,0.), (10,10,0))
        imageContrast = sampling.imagetransformer.ContrastChanger(0.9)
        transformers = [imageFlipper, imageRotator, imageShifter, imageContrast]

        for i in range(15):
            name = "crop_centered_0.9765_2_" + str(i) + ".nii"
            new_path = os.path.join(new_dir, name)
            if os.path.exists(new_path):
                os.remove(new_path)

        for i in range(15):
            crop_array = crop_array_orig
            for image_transformer in transformers:
                seed = i
                new_array = image_transformer.transform(crop_array, seed)
                new_array[new_array <= 0] = 0
                new_array = np.round(new_array)
                crop_array = new_array
            crop_array = crop_array.astype(np.int16)
            new_im = nib.Nifti1Image(crop_array, affine)
            name = "crop_centered_0.9765_2_" + str(i) + ".nii"
            new_path = os.path.join(new_dir, name)
            if os.path.exists(new_path):
                os.remove(new_path)
            nib.save(new_im, new_path)

def get_center_crop_CT(data):
    center_of_mass = np.round(measurements.center_of_mass(data))
    return center_of_mass

def remove_table(array):
    array = binary_opening(array, iterations=1).astype(np.uint8)
    # print("Opening")
    array = getLargestCC(array)
    # print("Largest")
    array = binary_dilation(array, iterations = 10).astype(np.uint8)
    # print("Dilation")
    array = binary_fill_holes(array).astype(np.uint8)
    # print("Holes 1")
    array = binary_fill_holes(array, structure= np.ones((5,5,5))).astype(np.uint8)
    # print("Holes 5")
    # array = binary_fill_holes(array, structure= np.ones((10,10,10))).astype(np.uint8)
    # print("Holes 10")
    return array

def findFirstSlice(array):
    i_sum = 0
    # i_mean = 0
    slice = array.shape[2] - 1
    # print(np.sum(array[...,slice]))
    while round(i_sum) <= 4096:
        i_sum = np.sum(array[..., slice])
        # print("Sum {}".format(i_sum))
        # i_mean = np.mean(array[...,slice])
        slice = slice - 1
        if slice == 0:
            print("No slice found")
            break
    first_slice = (array.shape[2] - 1 - slice)
    return int(first_slice)

def findFirstSlice_MR(array):
    # i_sum = 0
    i_mean = 0
    slice = array.shape[2] - 1
    # print(np.sum(array[...,slice]))
    while round(i_mean) <= 4:
        # i_sum = np.sum(array[...,slice])
        i_mean = np.mean(array[...,slice])
        print(i_mean)
        slice = slice - 1
    first_slice = (array.shape[2] - 1 - slice)
    return first_slice

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def segment_out_brain(data, threshold):
    thresh_data = data
    thresh_data[data < threshold] = 0
    sobel_data = np.hypot(filters.sobel(thresh_data, axis = 0),filters.sobel(thresh_data, axis = 1))
    sobel_data = morphology.binary_dilation(sobel_data, iterations = 5)
    sobel_data = morphology.binary_closing(sobel_data, iterations = 4)
    mask_brain = sobel_data.astype(np.int16)
    center_of_mass = measurements.center_of_mass(mask_brain)
    return mask_brain, center_of_mass

def crop_HN(im, center, voxel_sizes, size_x_mm, size_y_mm, size_z_mm, first_slice):
        im_array = im.get_fdata(caching = 'unchanged')
        im_array = im_array.astype(np.float32)
        size_x, size_y, size_z = im.shape
        # print('Initial shape of image {}'.format(im_array.shape))
        center_x = int(np.ceil(center[0]))
        center_y = int(np.ceil(center[1]))
        nb_pixels_x = np.ceil(size_x_mm / voxel_sizes[0])
        nb_pixels_y = np.ceil(size_y_mm / voxel_sizes[1])
        if size_z_mm:
            nb_pixels_z = int(np.ceil(size_z_mm / voxel_sizes[2]))
        # print('Nb of pixels we want in z {}'.format(nb_pixels_z))
        # print('First slice {}'.format(first_slice))
        #check if the crop would be outside the image
        window_x = int(np.ceil(nb_pixels_x / 2))
        window_y = int(np.ceil(nb_pixels_y / 2)) 
        needz = 0
        if (nb_pixels_z + first_slice) >= size_z:
            extra_pixels = nb_pixels_z + first_slice - size_z
            #last 2.5cm (~5slices) will be mirrored and then background = 0
            if extra_pixels <= 5:
                slice_array = im_array[:,:,:extra_pixels]
                bckg_ext = slice_array[:,:,::-1]
            elif extra_pixels > 5:
                needz = 1 
                slice_array = im_array[:,:,:5]
                slice_array = slice_array[:,:,::-1]
                slice_bckg = np.zeros((size_x, size_y, (extra_pixels - 5)))
                bckg_ext = np.concatenate((slice_bckg, slice_array), axis = 2)
            im_array = np.concatenate((bckg_ext, im_array), axis = -1)
        # print('After z bckg extenstion shape of image {}'.format(im_array.shape))
        size_z = im_array.shape[2]
        #check x
        if (center_x - window_x) < 0:
            bckg_ext = np.zeros((abs(center_x - window_x), size_y, size_z),dtype=np.float32)
            im_array = np.concatenate((bckg_ext, im_array), axis = 0)
            center_x = center_x + abs(center_x - window_x)
        elif (center_x + window_x) > size_x:
            bckg_ext = np.zeros((window_x - (size_x - center_x), size_y, size_z), dtype=np.float32)
            im_array = np.concatenate((im_array, bckg_ext), axis = 0)
        # print('After x bckg extenstion shape of image {}'.format(im_array.shape))
        size_x = im_array.shape[0]
        #check y       
        if (center_y - window_y) < 0:
            bckg_ext = np.zeros((size_x, abs(center_y - window_y), size_z), dtype=np.float32)
            im_array = np.concatenate((bckg_ext, im_array), axis = 1)
            center_y = center_y + abs(center_y - window_y)
        elif (center_y + window_y) > size_y:
            bckg_ext = np.zeros((size_x, window_y - (size_y - center_y), size_z), dtype=np.float32)
            im_array = np.concatenate((im_array, bckg_ext), axis = 1)
        # print('After y bckg extenstion shape of image {}'.format(im_array.shape))
        size_y = im_array.shape[1]
        im_new = nib.Nifti1Image(im_array, im.affine)
        # print("Ranges x {} - {}".format((center_x - window_x),(center_x + window_x)))
        # print("Ranges y {} - {}".format((center_y - window_y),(center_y + window_y)))
        # print("Ranges z {} - {}".format(-(nb_pixels_z + first_slice-1),-(first_slice-1)))
        crop = im_new.slicer[(center_x - window_x):(center_x + window_x),(center_y - window_y):(center_y + window_y),
               -(nb_pixels_z + first_slice-1):-(first_slice-1)]
        return crop, needz

def crop_HN_CT(im_array, center, voxel_sizes, size_x_mm = None, size_y_mm = None, size_z_mm = None, first_slice = 2):
        # im_array = im.get_fdata(caching = 'unchanged')
        im_array = im_array.astype(np.int16)
        size_x, size_y, size_z = im_array.shape

        # print('Initial shape of image {}'.format(im_array.shape))

        center_x = int(np.ceil(center[0]))
        center_y = int(np.ceil(center[1]))
        center_z = int(np.ceil(center[2]))

        nb_pixels_x = np.ceil(size_x_mm / voxel_sizes[0])
        nb_pixels_y = np.ceil(size_y_mm / voxel_sizes[1])
        if size_z_mm:
            nb_pixels_z = np.ceil(size_z_mm / voxel_sizes[2])
            window_z = int(np.ceil(nb_pixels_z / 2))
        #check if the crop would be outside the image
        window_x = int(np.ceil(nb_pixels_x / 2))
        window_y = int(np.ceil(nb_pixels_y / 2))

        # window = (window_x, window_y, window_z)
        # print(" Windows {}".format(window))
        needz = 0
        # print("Initial array shape {}".format(im_array.shape))
        if size_z_mm:
            if (center_z - window_z) < 0:
                # bckg_ext = -1024*np.ones((size_x, size_y, abs(center_z - window_z)),dtype=np.float32)
                bckg_ext = np.zeros((size_x, size_y, abs(center_z - window_z)), dtype=np.float32)
                im_array_new = np.concatenate((bckg_ext, im_array), axis = 2)
                center_z = center_z + abs(center_z - window_z)
                im_array = im_array_new.copy()
            elif (center_z + window_z) > size_z:
                # bckg_ext = -1024*np.ones((size_x, size_y, (center_z + window_z - size_z)), dtype=np.float32)
                bckg_ext = np.zeros((size_x, size_y, (center_z + window_z - size_z)), dtype=np.float32)
                im_array_new = np.concatenate((im_array, bckg_ext), axis = 2)
                im_array = im_array_new.copy()
            # print('After possible z bckg extenstion shape of image {}'.format(im_array.shape))
            size_z = im_array.shape[2]
        #check x
        if (center_x - window_x) < 0:
            # bckg_ext = -1024*np.ones((abs(center_x - window_x), size_y, size_z),dtype=np.float32)
            bckg_ext = np.zeros((abs(center_x - window_x), size_y, size_z), dtype=np.float32)
            im_array_new = np.concatenate((bckg_ext, im_array), axis = 0)
            im_array = im_array_new.copy()
            center_x = center_x + abs(center_x - window_x)
        if (center_x + window_x) > size_x:
            # bckg_ext = -1024*np.ones(((center_x + window_x - size_x), size_y, size_z), dtype=np.float32)
            bckg_ext = np.zeros(((center_x + window_x - size_x), size_y, size_z), dtype=np.float32)
            im_array_new = np.concatenate((im_array, bckg_ext), axis = 0)
            im_array = im_array_new.copy()
        # print('After possible x bckg extenstion shape of image {}'.format(im_array.shape))
        size_x = im_array.shape[0]
        #check y
        if (center_y - window_y) < 0:
            # bckg_ext = -1024*np.ones((size_x, abs(center_y - window_y), size_z), dtype=np.float32)
            bckg_ext = np.zeros((size_x, abs(center_y - window_y), size_z), dtype=np.float32)
            im_array_new = np.concatenate((bckg_ext, im_array), axis = 1)
            center_y = center_y + abs(center_y - window_y)
            im_array = im_array_new.copy()
        if (center_y + window_y) > size_y:
            # bckg_ext = -1024*np.ones((size_x, (center_y + window_y - size_y), size_z), dtype=np.float32)
            bckg_ext = np.zeros((size_x, (center_y + window_y - size_y), size_z), dtype=np.float32)
            im_array_new = np.concatenate((im_array, bckg_ext), axis = 1)
            im_array = im_array_new.copy()
        # print('After possible y bckg extenstion shape of image {}'.format(im_array.shape))
        size_y = im_array.shape[1]
        im_array = im_array.astype(np.int16)
        im_new = nib.Nifti1Image(im_array, np.eye(4))
        # print("Ranges x {} - {}".format((center_x - window_x),(center_x + window_x)))
        # print("Ranges y {} - {}".format((center_y - window_y),(center_y + window_y)))
        # print("Ranges z {} - {}".format((center_z - window_z),(center_z + window_z)))
        # crop = im_new.slicer[(center_x - window_x):(center_x + window_x),(center_y - window_y):(center_y + window_y),
        #        -(nb_pixels_z + first_slice-1):-(first_slice-1)]
        if size_z_mm:
            crop = im_new.slicer[(center_x - window_x):(center_x + window_x), (center_y - window_y):(center_y + window_y),
                  (center_z - window_z):(center_z + window_z)]
        else:
            crop = im_new.slicer[(center_x - window_x):(center_x + window_x), (center_y - window_y):(center_y + window_y),...]
        return crop, needz

def find_volume(array, axis = 2):
    sum = 0
    i_largest = 0
    for i in range(array.shape[2]):
        new_sum = array[:,:,i].sum()
        if new_sum >= sum:
            i_largest = i
            sum = new_sum
    if axis == 2:
        new_array = array[:,:,i_largest]
    elif axis == 1:
        new_array = array[:,i_largest,:]
    elif axis == 0:
        new_array = array[i_largest,...]
    return new_array


def get_seq_data(sequence, ignore_keys):
    seq_data = {}
    for seq in sequence:
        for s_key in seq.dir():
            if s_key in ignore_keys:
                continue
            s_val = getattr(seq, s_key, '')
            if type(s_val) == pydicom.sequence.Sequence:
                _seq = get_seq_data(s_val, ignore_keys)
                seq_data[s_key] = _seq
                continue
            if type(s_val) == str:
                s_val = format_string(s_val)
            else:
                s_val = assign_type(s_val, ignore_keys)
            if s_val:
                seq_data[s_key] = s_val
    return seq_data

def get_df_patient(path_patient, sample_image, path_nifti, 
                   tags, exclude_tags, dataset, slice_thickness):
    
    dicom_dict_tags = search_dicom_tags(sample_image, tags, exclude_tags)
    df = pd.DataFrame.from_dict(dict([(k,pd.Series(v)) for k,v in dicom_dict_tags.items()]))
    df_paths = pd.DataFrame({'PathPatient':[path_patient], 'PathNiftiFile':[path_nifti],
                             'Dataset': dataset, 'SliceThickness': slice_thickness})
    df_patient = pd.concat([df_paths, df], axis = 1, join = 'outer')
    return df_patient

def get_df_slice(path_patient, path_file, ds, tags, exclude_tags, dataset):
    dicom_dict_tags = search_dicom_tags(ds, tags, exclude_tags)
    df = pd.DataFrame.from_dict(dict([(k,pd.Series(v)) for k,v in dicom_dict_tags.items()]))          
    df_paths = pd.DataFrame({'PathPatient':[path_patient], 
                                     'PathImage': [path_file],
                                     'Dataset': dataset})
    df_all = pd.concat([df_paths, df], axis = 1, join = 'outer')
    return df_all

def get_suv_factor(sample_image):
    units = sample_image.Units
    error = []
    suv_factor = np.nan
    list_corrections = list(sample_image.CorrectedImage)
    RadiopharmaceuticalStartTime = str(sample_image.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)
    AcquisitionTime =  str(sample_image.AcquisitionTime)
    if  '.' in RadiopharmaceuticalStartTime:
        RadiopharmaceuticalStartTime = RadiopharmaceuticalStartTime.split(".")[0]
    if  '.' in AcquisitionTime:
        AcquisitionTime = AcquisitionTime.split(".")[0]
    if len(RadiopharmaceuticalStartTime)== 5:
        RadiopharmaceuticalStartTime = '0' + RadiopharmaceuticalStartTime
    if len(AcquisitionTime)== 5:
       AcquisitionTime = '0' + AcquisitionTime
    if 'DECY' in list_corrections:
        if units == 'BQML':
            try:
                if sample_image.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose != '':
                    dose = float(sample_image.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose) # in MBq or Bq??
                else:
                    dose = 0
                HL  = float(sample_image.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
                h_start = 3600*float(RadiopharmaceuticalStartTime[:2])
                h_stop = 3600* float(AcquisitionTime[:2])
                m_start = 60*float(RadiopharmaceuticalStartTime[2:4])
                m_stop = 60*float(AcquisitionTime[2:4])
                s_start = float(RadiopharmaceuticalStartTime[4:6])
                s_stop = float(AcquisitionTime[4:6])
                time2 = (h_stop+m_stop+s_stop-h_start-m_start-s_start)
                activity = dose*(2**(-time2/HL))
                try:
                    weight = float(sample_image.PatientWeight)*1000
                except:
                    print('Assignig weight of 70kg')
                    weight = 70000.
                suv_factor = weight/activity
            except AttributeError:
                error = 'Attribute to calculate SUV missing'
        elif units == 'GML':
            suv_factor = 1.
        elif units == 'CNTS':
            weight = 1.
            try:
                suv_factor = float(sample_image[0x7053,0x1000].value)
            except AttributeError: 
                try:
                    suv_factor = float(sample_image[0x7053,0x1009].value) * float(sample_image.RescaleSlope)
                except:
                    error = "Not possible to determine SUV factor - Scan in counts units - Philips tags not accessible"
    else:
        error = 'Units not known'

    return suv_factor, error

def sort_slices(files_dcm, path_patient):
    error = []
    positions = []
    try:
        files_dcm.sort(key = lambda x: dcmread(os.path.join(path_patient,x),force=True).ImagePositionPatient[2], reverse = True)
        datasets = [dcmread(os.path.join(path_patient,x),force=True) for x in files_dcm]
        positions = [round(float(ds.ImagePositionPatient[2]),2) for ds in datasets]
        positions.sort(reverse = False)
        # positions = [round(float(ds.ImagePositionPatient[2]),3) for ds in datasets]
    except AttributeError: 
        try:
            files_dcm.sort(key = lambda x: dcmread(os.path.join(path_patient,x),force=True).SliceLocation, reverse = True)
        except AttributeError:
            try:
                sample_image = dcmread(os.path.join(path_patient, files_dcm[0],force=True))
                if sample_image.PatientPosition == 'HFS':
                    files_dcm.sort(key = lambda x: dcmread(os.path.join(path_patient,x),force=True).ImageIndex, reverse = True)
                if sample_image.PatientPosition == 'FFS':
                    files_dcm.sort(key = lambda x: dcmread(os.path.join(path_patient,x),force=True).ImageIndex)
            except AttributeError:
                error = 'Ordering of slices not possible due to lack of attributes'
    return files_dcm, positions, error

def normalizeSUV(ds, nb, UID, suv_factor):
    error = []
    try:
        ds.FrameOfReferenceUID = ds.FrameOfReferenceUID[:2]+UID #change UID so it is treated as a new image
        ds.SeriesInstanceUID = ds.SeriesInstanceUID[:2]+UID   
        ds.SOPInstanceUID = ds.SeriesInstanceUID[:2]+UID+str(nb)
        data_array = ds.pixel_array
        #Check the data type
        bitsRead = str(ds.BitsAllocated)
        sign = int(ds.PixelRepresentation)
        if sign == 1:
            bitsRead = 'int'+bitsRead
        elif sign ==0:
            bitsRead = 'uint'+bitsRead  
        data_array = data_array.astype(dtype=bitsRead) #make sure it is the correct data type
        data16 = (data_array * ds.RescaleSlope + ds.RescaleIntercept) * suv_factor
        data16 = np.array(np.around(data16), dtype=bitsRead)
        ds.RescaleSlope = 1
        ds.RescaleIntercept = 0
        ds.PixelData = data16.tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = True
    except AttributeError:
        error = 'Could not get pixel array'
    return ds, error

def normalizeCT(ds, nb, UID):
    error = []
    # try:
    ds.FrameOfReferenceUID = ds.FrameOfReferenceUID[:2]+UID #change UID so it is treated as a new image
    ds.SeriesInstanceUID = ds.SeriesInstanceUID[:2]+UID
    ds.SOPInstanceUID = ds.SeriesInstanceUID[:2]+UID+str(nb)
    data_array = ds.pixel_array
    data_array[data_array == -2000] = 0
    intercept = ds.RescaleIntercept
    intercept = np.array(intercept, dtype = 'int16')

    slope = ds.RescaleSlope
    #Check the data type
    bitsRead = str(ds.BitsAllocated)
    sign = int(ds.PixelRepresentation)
    if sign == 1:
        bitsRead = 'int'+bitsRead
        data_array = data_array.astype(dtype=bitsRead)
    elif sign == 0:
        bitsRead = 'uint'+bitsRead
        if intercept < 0:
            data_array = data_array.astype('int16')
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.PixelRepresentation = 1
    if slope != 1:
        print('Slope different than one')
        data_array = slope * data_array
        data_array = data_array.astype('int16')
    if intercept != 0:
        # print(intercept)
        # print(np.min(data_array))
        data_array += intercept
    # data_array[data_array < -1024] = -1024
    # data_array[data_array > 3071] = 3071 #everything above this value is artifact
    # print("Minimum data 16 {}".format(np.min(data_array)))
    # print("Maximum data 16 {}".format(np.max(data_array)))
    data_array[data_array < -1024] = -1024
    data_array[data_array > 3071] = 3071

    data16 = data_array.astype('int16')
    # print("Minimum data 16 {}".format(np.min(data16)))
    # print("Maximum data 16 {}".format(np.max(data16)))
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.PixelData = data_array.tobytes()
    # ds.is_little_endian = True
    ds.is_implicit_VR = True
    # print(ds.is_little_endian)
    # print(ds.is_implicit_VR)
    # except AttributeError:
    #     error = 'Could not get pixel array'
    return ds, error

def format_string(in_string):
    formatted = re.sub(r'[^\x00-\x7f]', r'', str(in_string))  # Remove non-ascii characters
    formatted = ''.join(filter(lambda x: x in string.printable, formatted))
    if len(formatted) == 1 and formatted == '?':
        formatted = None
    return formatted

def assign_type(s, ignore_keys):
    if type(s) == list or type(s) == pydicom.multival.MultiValue:
        try:
            for x in s:
                if type(x) == pydicom.valuerep.DSfloat:
                    list_values = [float(x) for x in s if x not in ignore_keys]
                    return str(tuple(list_values))
                else:
                    list_values = [x for x in s if x not in ignore_keys]
                    return str(tuple(list_values))
        except ValueError:
            try:
                return [float(x) for x in s if x not in ignore_keys]
            except ValueError:
                return [format_string(x) for x in s if ((len(x) > 0) and (x not in ignore_keys))]
    elif type(s) == pydicom.sequence.Sequence:
            return get_seq_data(s,ignore_keys)      
    else:
        s = str(s)
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return format_string(s)

def get_mask2(contours, z, Pat_position, pix_spacing, mask):
    pos_r = Pat_position[1]
    spacing_r = pix_spacing[1]
    pos_c = Pat_position[0]
    spacing_c = pix_spacing[0]
    label=mask
    error = 0
    # print("Z {}".format(z))
    for c in contours:
        c_array = np.asarray(c)
        nodes = c_array.reshape((-1, 3))
        # print("Nodes {}".format(nodes))
        #assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
        try:
            z_index = z.index(nodes[0, 2])
        except:
            try:
                c = [round(element,1) for element in c]
                c_array = np.asarray(c)
                nodes = c_array.reshape((-1, 3))
                z = [round(element,1) for element in z]
                z_index = z.index(nodes[0,2])
            except:
                try:
                    c = [round(element) for element in c]
                    c_array = np.asarray(c)
                    nodes = c_array.reshape((-1, 3))
                    z = [round(element) for element in z]
                    z_index = z.index(nodes[0,2])
                except:
                    error = 1
                    print(z)
                    print(c_array)
                    print("Error")
        # print(z_index)
        if not error:
            r = (nodes[:, 1] - pos_r) / spacing_r
            c = (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            label[rr, cc, z_index] = 1
    return label


def find_info(subdir):
    os.chdir(subdir)
    try:
        info = glob.glob('*.{}'.format('csv'))
        info = ''.join(info)
        info_patients = pd.read_csv(info)
        return info_patients
    except:
        try:
            info = glob.glob('*.{}'.format('xlsx'))
            info = ''.join(info)
            info_patients = pd.read_excel(info)
            return info_patients
        except:
            print("No metadata file found in {}".format(subdir))

def build_metadata(metadata_tags, info_patients):
    count_csv = 0
    for tag in metadata_tags:
        try:
            if count_csv == 0:
                metadata_tag = info_patients[[tag]]
                metadata = metadata_tag.copy()
                count_csv = 1
            else:
                metadata = pd.concat([metadata, info_patients[[tag]]], axis = 1, sort = True)
                count_csv += 1
        except:
            print('Tag not in info csv file {}'.format(tag))
            if count_csv == 0:
                metadata_tag= pd.DataFrame([np.nan] * len(info_patients.index), columns = [tag])
                metadata = metadata_tag.copy()
                count_csv = 1
            else:
                metadata[tag] = np.nan
                count_csv += 1
    return metadata
def load_metadata(csvfile, df_final, subdir):
    df_new = pd.read_csv(os.path.join(subdir,csvfile))
    for name in df_final.columns:
        if name in df_new.columns:
            df_final[name] = df_new[name]
    return df_final

def find_excel(subdir):
    os.chdir(subdir)
    info = glob.glob('*.{}'.format('xlsx'))
    print(info)
    return info
    
def search_dicom_tags(ds, tags, ignore_tags):
    dict_dicom = {}
    for name in tags:
        if ((name in ds) and (name not in ignore_tags)):
            dict_dicom[name] = assign_type(ds[name].value, ignore_tags)
        else: 
            dict_dicom[name] = 'NaN'
    delete_keys = []
    dict_dicom2 = dict_dicom.copy()
    for key, value in dict_dicom.items():
        if type(value) == dict:
            out_dict = dict_dicom[key]
            dict_dicom2.update(out_dict) 
            delete_keys.append(key)
    for key in delete_keys:
        del dict_dicom2[key]
    return dict_dicom2 

def get_slice_thickness(path_patient, slice1, slice2):
    slice_thickness = np.nan
    dc1 = dcmread(os.path.join(path_patient,slice1),force=True)
    dc2 = dcmread(os.path.join(path_patient,slice2),force=True)
    error = []
    try:
        slice_thickness = np.abs(dc1.ImagePositionPatient[2] - dc2.ImagePositionPatient[2])
    except AttributeError:
        try:
            slice_thickness = np.abs(dc1.SliceLocation - dc2.SliceLocation)
        except:
            try:
                slice_thickness = dc1.SliceThickness
            except AttributeError:
                slice_thickness = 0
                error = 'Could not determine slice thickness'

    return slice_thickness, error

def get_pixels(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    image = image.astype(np.int16)
    return np.array(image, dtype=np.int16)

