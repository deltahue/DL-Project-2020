# DL-Project-2020

### Getting started

- Clone this repo:
```bash
git clone https://github.com/taesungp/contrastive-unpaired-translation CUT
cd CUT
```

- Dowload the Dataset


- Train the CUT model:
```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT
```

 Train the FastCUT model
 ```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_FastCUT --CUT_mode FastCUT
```

 Train the CycleGAN model
 ```bash
python tr

The checkpoints will be stored at `./checkpoints/grumpycat_*/web`.


### CUT, FastCUT, and CycleGAN
<img src="imgs/horse2zebra_comparison.jpg" width="800px"/><br>

CUT is trained with the identity preservation loss and with `lambda_NCE=1`, while FastCUT is trained without the identity loss but with higher `lambda_NCE=10.0`. Compared to CycleGAN, CUT learns to perform more powerful distribution matching, while FastCUT is designed as a lighter (half the GPU memory, can fit a larger image), and faster (twice faster to train) alternative to CycleGAN. Please refer to the [paper](https://arxiv.org/abs/2007.15651) for more details.

- Test the CUT model:
```bash
python test.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT --phase train
```
