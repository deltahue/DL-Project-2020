# DL-Project-2020

The following is a description to reproduce our training and testing methods.

## Clone our repository
- Clone this repository:
```bash
git clone https://github.com/taesungp/contrastive-unpaired-translation CUT
cd CUT
```

## Dowload the dataset
Dowload the medical dataset that was submitted together with the report and store it under the folder ./datasets/

## Train the models
### CUT
 ```bash
!python -u train.py --dataroot '/datasets/train/train' --name experiment_CUT --batch_size 1 --max_dataset_size 50  --output_nc 1 --input_nc 1 --verbose --lr 0.0001 --preprocess none --crop_size 256 --model cut --lr 0.0001 --preprocess none --save_epoch_freq 1 --verbose --display_id 0 --continue_train --epoch_count 12
 ```
 The checkpoints will be stored at `./checkpoints/experiment_CUT`.


### FastCUT
 ```bash
!python -u train.py --dataroot '/datasets/train/train' --name experiment_FastCUT  --batch_size 1 --max_dataset_size 50  --output_nc 1 --input_nc 1 --verbose --lr 0.0001 --preprocess none --crop_size 256 --model cut --mode FastCUT --lr 0.0001 --preprocess none --save_epoch_freq 1 --verbose --display_id 0 --continue_train --epoch_count 12
 ```
 The checkpoints will be stored at `./checkpoints/experiment_FastCUT`.

### CycleGAN
 ```
!python -u train.py --dataroot '/datasets/train/train' --name experiment_CycleGAN --batch_size 1 --max_dataset_size 50  --output_nc 1 --input_nc 1 --verbose --lr 0.0001 --preprocess none --crop_size 256 --model cycle_gan --lr 0.0001 --preprocess none --save_epoch_freq 1 --verbose --display_id 0 --continue_train --epoch_count 12
 ```
 The checkpoints will be stored at `./checkpoints/experiment_CycleGAN`.


### CUT, FastCUT, and CycleGAN
CUT is trained with the identity preservation loss and with `lambda_NCE=1`, while FastCUT is trained without the identity loss but with higher `lambda_NCE=10.0`. Compared to CycleGAN, CUT learns to perform more powerful distribution matching, while FastCUT is designed as a lighter (half the GPU memory, can fit a larger image), and faster (twice faster to train) 
alternative to CycleGAN. Please refer to the [paper](https://arxiv.org/abs/2007.15651) for more details.

## Test the models
To obtain the synthetic CT images, the steps to reproduce are the following:

### CUT

- Test the CUT model:
```bash
  python test.py --dataroot "/datasets/test/test" --name experiment_CUT --crop_size 512 --load_size 512 --num_threads 1 --no_flip  --output_nc 1 --input_nc 1  --model cut
```

### FastCUT

```bash
  python test.py --dataroot "/datasets/test/test" --name experiment_FastCUT --crop_size 512 --load_size 512 --num_threads 1 --no_flip  --output_nc 1 --input_nc 1  --model cut --mode FastCUT
```

### CycleGAN
```bash
  python test.py --dataroot "/content/dataset/test/test"--name experiment_CycleGAN --crop_size 512 --load_size 512 --num_threads 1 --no_flip  --output_nc 1 --input_nc 1  --model cycle_gan```
```
# How to calculate the metrics

The metrics can be calculated using the script `metrics-evaluation.py`. The script can be called the following way:
```bash
python evaluate_metrics.py --bodymask_path '/masks/' --fake_slices_path 'path_to_generated_slices' --real_slices_path 'path_to_real_slices' --FID
```

Furthermore there is an option to geht some debug image information with `--debug_images` and the path where the metrics are saved can be specified using the argument `--results_path` (default is in the current directory).

The `--FID` tag can be used to perform Frechet Inception Distance calculations. For this the modified pip package needs to be cloned from 'https://github.com/deltahue/pytorch-fid' and installed locally.

