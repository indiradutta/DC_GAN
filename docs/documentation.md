## Parameters

Parameter |  &nbsp;&nbsp;&nbsp;&nbsp; Value &nbsp;&nbsp;&nbsp;&nbsp; |
:------------: | :---: |
batch_size | 128 |
image_size | 64 |
nc | 3 |
nz | 100 |
ngf | 64 |
ndf | 64 |
num_epochs | 5 |
lr | 0.0002 |
beta1 | 0.5 |
ngpu | 1 |

## Result Documentation
After running *DCGAN* on the CelebA Dataset for 5 epochs on GPU (computationally very expensive) we got the following output images along with the Generator and Discriminator losses.

## Batch of images from the Generator after 5 epochs 
<img src="/results/result2.png">

## Losses after each epoch
No. of Epochs | Generator Loss | Discriminator Loss |
:------------: | :------------: | :------------: |
1 | 0.7894 | 1.0838 |
2 | 0.7277 | 1.0489 |
3 | 0.7796 | 0.9256 |
4 | 0.6330 | 1.1345 |
5 | 0.7519 | 1.0138 |

## Plot for Generator Loss and Discriminator Loss w.r.t number of iterations
<img src="/results/losses.png">

