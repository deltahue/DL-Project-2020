# DL-Project-2020


### Clone our repository
- Clone this repository:
```bash
git clone https://github.com/taesungp/contrastive-unpaired-translation CUT
cd CUT
```

### Dowload the dataset
Dowload the medical dataset that was submitted together with the report and store it under the folder ./datasets/

### Train the models
# CUT
 ```bash
!python -u train.py --dataroot '/content/dataset/train' --name experiment_CUT --batch_size 1 --max_dataset_size 50  --output_nc 1 --input_nc 1 --verbose --lr 0.0001 --preprocess none --crop_size 256 --model cut --lr 0.0001 --preprocess none --save_epoch_freq 1 --verbose --display_id 0 --continue_train --epoch_count 12
 ```

# FastCUT
 ```bash
!python -u train.py --dataroot '/content/dataset/train' --name experiment_FastCUT  --batch_size 1 --max_dataset_size 50  --output_nc 1 --input_nc 1 --verbose --lr 0.0001 --preprocess none --crop_size 256 --model cut --mode FastCUT --lr 0.0001 --preprocess none --save_epoch_freq 1 --verbose --display_id 0 --continue_train --epoch_count 12
 ```

# CycleGAN
 ```
!python -u train.py --dataroot '/content/dataset/train' --name experiment_CycleGAN --batch_size 1 --max_dataset_size 50  --output_nc 1 --input_nc 1 --verbose --lr 0.0001 --preprocess none --crop_size 256 --model cycle_gan --lr 0.0001 --preprocess none --save_epoch_freq 1 --verbose --display_id 0 --continue_train --epoch_count 12
 ```

The checkpoints will be stored at `./checkpoints/grumpycat_*/web`.


### CUT, FastCUT, and CycleGAN
CUT is trained with the identity preservation loss and with `lambda_NCE=1`, while FastCUT is trained without the identity loss but with higher `lambda_NCE=10.0`. Compared to CycleGAN, CUT learns to perform more powerful distribution matching, while FastCUT is designed as a lighter (half the GPU memory, can fit a larger image), and faster (twice faster to train) 
alternative to CycleGAN. Please refer to the [paper](https://arxiv.org/abs/2007.15651) for more details.

### Test the models
To obtain the synthetic CT images, the steps to reproduce are the following:

- Test the CUT model:
```bash
  python test.py --dataroot "/content/dataset/test/test"  --crop_size 512 --load_size 512 --num_threads 1 --no_flip --output_nc 1 --input_nc 1  
```


```bash
  python test.py --dataroot "/content/dataset/test/test"  --crop_size 512 --load_size 512 --num_threads 1 --no_flip --output_nc 1 --input_nc 1  
```

```bash
python test.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT --phase train
```



