import os
import pandas as pd
import shutil
import numpy as np

from pydicom import dcmread, read_file

import os
import numpy as np
import pandas as pd
from readpatient import ReadCTSeries
from helpers import find_info, build_metadata, get_sequences, get_slice_thickness, search_dicom_tags, sort_slices
import nibabel as nib
import glob
import shutil
import dicom2nifti
from pydicom import dcmread, read_file
import dicom2nifti.settings as settings
import numpy as np
import nibabel as nib
from nibabel import processing
from helpers import findFirstSlice, crop_HN_CT, get_center_crop_CT, remove_table
import pandas as pd
from scipy.ndimage import morphology, measurements, filters, \
    binary_closing, binary_opening, binary_erosion, binary_dilation, binary_fill_holes
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
import skimage
import net
global path_nifti
settings.disable_validate_slice_increment()
settings.disable_validate_orthogonal()

path_all = "/srv/beegfs02/scratch/headneck_lung_cancer/data/data/HN_MRIdian_final"
path_nifti = "/srv/beegfs02/scratch/headneck_lung_cancer/data/data/HN_MRIdian_nifti/"
path_3D_nifti =  "/srv/beegfs02/scratch/headneck_lung_cancer/data/data/HN_MRIdian_crops2/"
path_2D_nifti =  "/srv/beegfs02/scratch/headneck_lung_cancer/data/data/HN_MRIdian_crops2_axialslices/"
os.chdir(path_all) #make it current directory

#Settings
createNifti = True
count_patient = 0
new_max = 0
for subdir, dirs, files in os.walk(path_all):
    dirs.sort(key = lambda x: os.path.getmtime(os.path.join(subdir, x)), reverse = True) #order directories based on the time of their creation
    if not os.path.isdir(subdir):
        continue
    if subdir != path_all:
        dcm_files = [file for file in files if file.endswith("dcm")]
        if dcm_files == []:
            continue
        _, dir_patient_series = subdir.split(path_all)
        _, patient, series = dir_patient_series.split("/")
        name_dir = []
        if "mrsim" in series.lower():
            name_dir ="MR"
        elif "CT" in series:
            name_dir = "CT"
        elif "Fr" in series:
            name_dir ="MR"

        if (patient == "PAT1") or (patient == "PAT3") or (patient == "PAT5"):
            set = "test"
        else:
            set = "train"

        new_nifti_dir = os.path.join(os.path.join(os.path.join(path_nifti, name_dir ), patient), series)
        print(new_nifti_dir)
        if not os.path.exists(new_nifti_dir):
            os.makedirs(new_nifti_dir)
        for dcm_file in dcm_files:
            old_name = os.path.join(subdir, dcm_file)
            if name_dir != "CT_registered":
                try:
                    ds = dcmread(old_name)
                    ds[0x0018,0x0080].value = 0.
                    ds[0x0018,0x0081].value = 0.
                    ds[0x0018,0x0091].value = 0.
                    ds.save_as(old_name)
                except:
                    pass
            new_name = os.path.join(new_dicom_dir,dcm_file)
            shutil.copy(old_name, new_name)
            new_nifti_name = os.path.join(new_nifti_dir, dcm_file)
        dicom2nifti.convert_directory(subdir, new_nifti_dir, compression=False, reorient=True)
        image_file = [file for file in os.listdir(new_nifti_dir) if ((file.endswith('.nii')) and (not file.startswith("body")) and (
                not file.startswith("mask")))]
        if image_file == []:
            continue
        image = nib.load(os.path.join(new_nifti_dir, image_file[0]))
        _, newdir = new_nifti_dir.split(path_nifti)
        print(newdir)
        newdir_crop = os.path.join(path_crops, newdir)
        print(newdir_crop)
        if not os.path.exists(newdir_crop):
            os.makedirs(newdir_crop)
        image = processing.resample_to_output(image, voxel_sizes=[1., 1., 1.], order=1, mode='constant', cval=0)
        mr_image = processing.resample_to_output(image, voxel_sizes=[1., 1., 1.], order=1, mode='constant', cval=0)
        image_data = image.get_fdata(caching='unchanged')
        mr_image_data = mr_image.get_fdata(caching='unchanged')
        print("Resized shape {}".format(image_data.shape))
        body_mask = np.zeros(shape=mr_image_data.shape)
        body_mask2 = np.zeros(shape=image_data.shape)

        if modality == "MR":
            print("mr")
            body_mask[mr_image_data > 20] = 1
        elif modality == "CT":
            print("ct")
            body_mask2[image_data > -100] = 1
            body_mask2[image_data > 200] = 0
            body_mask2[..., :60] = 0
            body_mask2 = remove_table(body_mask2)
            body_mask2 = binary_dilation(body_mask2, iterations = 2).astype(np.int16)

        first_slice = np.maximum(findFirstSlice(body_mask), 2)
        print("First slice {}".format(first_slice))

        masked_image = body_mask2*image_data
        # body_mask_im = nib.Nifti1Image(body_mask, image.affine)
        masked_im = nib.Nifti1Image(masked_image, image.affine)
        modality = "CT"
        if modality == "MR":
            masked_image[body_mask == 0] = 0
        elif modality == "CT":
            masked_image[body_mask2 == 0] = -1024

        nb_pixels_z = 300
        crop_z = masked_im.slicer[..., -(nb_pixels_z + first_slice):-first_slice]
        crop_z_data = crop_z.get_fdata(caching = 'unchanged')
        center_of_crop = get_center_crop_CT(body_mask[..., -150:])
        print(center_of_crop)

        voxel_sizes = [1., 1., 1.]
        size_x_mm, size_y_mm, size_z_mm = [512, 512, 300]
        shape_data = image_data.shape

        crop, needz = crop_HN_CT(crop_z_data, center_of_crop, voxel_sizes, size_x_mm = size_x_mm, size_y_mm = size_y_mm, first_slice = first_slice)
        path_crop = os.path.join(newdir_crop, "crop.nii")
        if os.path.exists(path_crop):
            os.remove(path_crop)
        crop_data = crop.get_fdata(caching="unchanged")
        crop_data = crop_data.astype(np.int16)
        print("Shape crop {}".format(crop_data.shape))
        crop = nib.Nifti1Image(crop_data, crop.affine)
        nib.save(crop, path_crop)

        newdir_crop2 = os.path.join(os.path.join(path_2D_nifti, set), modality)
        if not os.path.exists(newdir_crop2):
            os.makedirs(newdir_crop2)
        for i in range(crop_data.shape[2]):
            im = crop.slicer[...,i:(i+1)]
            path_nifti_slice = os.path.join(newdir_crop2, patient + "-" + series + "-" + str(i) + ".nii")
            im_data = im.get_fdata(caching="unchanged")
            im_data = im_data.astype(np.int16)
            # if not np.any(crop_data):
            #     continue
            if os.path.exists(path_nifti_slice):
                os.remove(path_nifti_slice)
            print(path_nifti_slice)
            im = nib.Nifti1Image(im_data, im.affine)
            nib.save(im, path_nifti_slice)
