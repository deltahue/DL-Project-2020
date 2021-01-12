import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import nibabel as nib
import numpy as np
from patchify import patchify, unpatchify
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torchio as tio
from torchio import RandomElasticDeformation
import torch
import time

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        start_time = time.time()
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # print("A path {}".format(A_path))
        pat, mr_series, slicenii = os.path.basename(A_path).split("-")
        if len(slicenii) == 5:
            slice_number = int(slicenii[:1])
        elif len(slicenii) == 6:
            slice_number = int(slicenii[:2])
        elif len(slicenii) == 7:
            slice_number = int(slicenii[:3])
        B_path = '/path/to/B'
        # print(slice_number)


        slice_number_orig = slice_number
        r = 5
        if slice_number <= 10:
            slice_number += random.randint(0, 5)
        elif slice_number >= 290:
            slice_number += random.randint(-5,0)
        else:
            slice_number += random.randint(-5, 5)
        for i in range(150):
            if not os.path.exists(B_path):
                # print(slice_number)
                slicenii2 = str(slice_number) + ".nii"
                pat_ct_slice = pat + "-CTSim-" + slicenii2
                # print(pat_ct_slice)
                B_paths = [path_ct for path_ct in self.B_paths if pat_ct_slice in path_ct]
            if B_paths == []:
                # print("empty")
                slice_number = int(slice_number)
                # print(slice_number)
                if slice_number < 61:
                    slice_number += 1
                elif slice_number > 260:
                    slice_number += -1
            else:
                B_path = B_paths[0]
                try:
                    B_img_nifti = nib.load(B_path)
                    B_img_numpy = B_img_nifti.get_fdata(caching = "unchanged")
                    if not np.any(B_img_numpy):
                        print("Not any {}".format(B_path))
                        B_path = '/path/to/B'
                        slice_number = int(slice_number)
                        # print(slice_number)
                        if slice_number < 61:
                            slice_number += 1
                        elif slice_number > 260:
                            slice_number += -1
                    else:
                        break
                    # slice_number = slice_number_orig
                except:
                    slice_number = int(slice_number)
                    # print(slice_number)
                    if slice_number < 61:
                        slice_number += 1
                    elif slice_number > 260:
                        slice_number += -1
                    pass

            # r += 1
        # if self.opt.serial_batches:   # make sure index is within then range
        #     index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
        #     index_B = random.randint(0, self.B_size - 1)

        # B_path = self.B_paths[index_B]
        #A_img = Image.open(A_path).convert('RGB')
        # print("B_path {}".format(B_path))
        # print("Time {}".format(time.time() - start_time))
        A_img_nifti = nib.load(A_path)
        A_img_numpy = A_img_nifti.get_fdata(caching = "unchanged")
        A_img_numpy = np.squeeze(A_img_numpy)
        # A_img_numpy = np.rot90(A_img_numpy)
        A_img_numpy = (A_img_numpy - np.amin(A_img_numpy))/(np.amax(A_img_numpy) - np.amin(A_img_numpy)) #Normalize MR to be in range [0, 255]
        if np.amax(A_img_numpy) == 0:
            print("MR empty in {}".format(A_path))
        A_img_numpy[A_img_numpy > 1.] = 1.
        A_img_numpy[A_img_numpy < 0.] = 0.
        A_img_numpy = 255*A_img_numpy
        A_nonzero = np.count_nonzero(A_img_numpy)
        A_img_numpy = A_img_numpy.astype(np.uint8)
        A_img = Image.fromarray(A_img_numpy)

        if self.opt.isTrain:

            if self.opt.crop_size == 256:
                # Random crop
                i, j, h, w = transforms.RandomCrop.get_params(A_img, output_size=(self.opt.crop_size,self.opt.crop_size))
                A_crop = TF.crop(A_img, i, j, h, w)
                for i in range(50):
                    if not A_crop.getbbox():
                        i, j, h, w = transforms.RandomCrop.get_params(A_img, output_size=(self.opt.crop_size, self.opt.crop_size))
                        A_crop = TF.crop(A_img, i, j, h, w)
                    else:
                        break
                np_im = np.array(A_crop)
                for i in range(100):
                    if np.count_nonzero(np_im) < round(A_nonzero // 3):
                        i, j, h, w = transforms.RandomCrop.get_params(A_img, output_size=(self.opt.crop_size, self.opt.crop_size))
                        A_crop = TF.crop(A_img, i, j, h, w)
                        np_im = np.array(A_crop)
            else:
                A_crop = A_img
        #B_img = Image.open(B_path).convert('RGB')
        # B_img_nifti = nib.load(B_path)
        # B_img_numpy = B_img_nifti.get_fdata(caching = "unchanged")
        try:
            B_img_numpy = np.squeeze(B_img_numpy)
        except:
            print("Error {}".format(A_path))
            print("Error {}".format(B_path))
        # B_img_numpy = np.rot90(B_img_numpy)
        B_img_numpy = (B_img_numpy + 1024.)/4095. #Normalize CT to be in range [0, 255]   
        B_img_numpy[B_img_numpy > 1.] = 1.
        B_img_numpy[B_img_numpy < 0.] = 0.
        if np.amax(B_img_numpy) == 0:
            print(B_path)
        B_img_numpy = 255*B_img_numpy
        B_img_numpy = B_img_numpy.astype(np.uint8)
        B_img = Image.fromarray(B_img_numpy)
        if self.opt.isTrain:
            if self.opt.crop_size == 256:
                B_crop = TF.crop(B_img, i, j, h, w)
            else:
                B_crop = B_img
        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
            is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
            modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
            transform = get_transform(modified_opt)
            A = transform(A_crop)
            np_a = np.array(A)
            # print("A: max {}, min {}".format(np.amax(np_a),np.amin(np_a)))
            B = transform(B_crop)
            np_b = np.array(B)
            # print("B: max {}, min {}".format(np.amax(np_b),np.amin(np_b)))
            if np.amax(np_b) < 0:
                print(B_path)

        else:
            transform = transforms.ToTensor()
            A = transform(A_img)
            B = transform(A_img)
        # A = torch.unsqueeze(A, dim = 3)
        # B = torch.unsqueeze(B, dim = 3)
        # el_def = RandomElasticDeformation(num_control_points=6, locked_borders=2)
        # A = el_def(A)
        # B = el_def(B)
        # A = torch.squeeze(A, dim = 3)
        # B = torch.squeeze(B, dim = 3)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
