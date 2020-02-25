# SPULTRA: Low-Dose CT Image Reconstruction with Joint Statistical and Learned Image Models

IEEE Xplore: https://ieeexplore.ieee.org/document/8794829/

Arxiv: https://arxiv.org/abs/1808.08791


# Data:

We provide data for the 3D XCAT phantom simulations in the folder "data". The Matlab code to generate data is "axial_proj_data_maker.m".

We also provide initial PWLS-EP images for PWLS-ULTRA and SPULTRA in the folder "pwls_ep". The pre-learned transform is saved in the "transform_ost" folder. One can reproduce the pre-learned 3D transform using the code "OCTOBOS_LearnTransform3D.m".

Reconstruction results of SPULTRA and PWLS-ULTRA shown in the paper can be reproduced with "main_*.m".


# Implementation:

The codes should be run with Michigan Image Reconstruction Toolbox (MIRT).
http://web.eecs.umich.edu/~fessler/code/index.html


# Reference: 

[1] W. P. Segars, M. Mahesh, T. J. Beck, E. C. Frey, and B. M. W. Tsui, “Realistic CT simulation using the 4D XCAT phantom,” Med. Phys., vol. 35, no. 8, pp. 3800–3808, Aug. 2008.

[2] X. Zheng, S. Ravishankar, Y. Long, and J. A. Fessler, “PWLS-ULTRA: An efficient clustering and learning-based approach for low-dose 3D CT image reconstruction,” IEEE Trans. Med. Imag., vol. 37, no. 6, pp.1498–1510, 2018.
# SPULTRA-for-Low-Dose-CT-Image-Reconstruction
