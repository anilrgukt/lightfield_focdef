# Light Field Reconstruction from Coded images 
- Official Tensorflow implementation for our [TCI paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8878019) on A Unified Learning-Based Framework for Light Field Reconstruction From Coded Projections

# Demo Video
[![Watch the video](https://i.imgur.com/SCpwnAU.png)](https://youtu.be/dVxvcEwRS_U)

# Prerequisites
- NVIDIA GPU + CUDA + CuDNN
- Python 3.6
- tensorflow-gpu
- Other Python dependencies: numpy, scipy, scikit-image, imageio, matplotlib

# Datasets
Kalantari et al.: http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/

Flowers dataset: https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ICCV17/lfsyn/LF_Flowers_Dataset.tar.gz

# Training:
Run nbs/ulf_focdef_train.ipynb

# Testing:
Run nbs/ulf_focdef_test.ipynb

# Pretrained models:
https://drive.google.com/drive/folders/1Ja2bFg9ThpjgRHw2Wg0gNubdKvOlJA6q?usp=sharing

# Citation
```
@ARTICLE{8878019, 
         author={A. K. {Vadathya} and S. {Girish} and K. {Mitra}}, 
         journal={IEEE Transactions on Computational Imaging}, 
         title={A Unified Learning-Based Framework for Light Field Reconstruction From Coded Projections}, 
         year={2020}, 
         volume={6}, 
         number={}, 
         pages={304-316},
}
```
