# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Network (DCGAN) implementation
This is an implementation of the research paper <a href = "https://arxiv.org/abs/1511.06434.pdf">"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"</a> written by Alec Radford, Luke Metz, Soumith Chintala.

## Dataset
The original paper had used three datasets for training the DCGAN namely - *Large-scale Scene Understanding (LSUN) (Yu et al., 2015), Imagenet-1k and a newly assembled Faces dataset*. However due to computational and other limitations, we have used <a href = "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html">Large-scale CelebFaces Attributes (CelebA) Dataset</a>.

### Citation
``` 
@inproceedings{liu2015faceattributes,
 title = {Deep Learning Face Attributes in the Wild},
 author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
 booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
 month = {December},
 year = {2015} 
}
``` 
### Guidelines to download, setup and use the dataset
The CelebA dataset may be downloaded <a href = "https://drive.google.com/file/d/1yW6QkWcd6sWYB2rw9d-A36woiXVLTpny/view?usp=sharing">here</a> as a file named *img_align_celeba.zip*. 

**Please write the following commands on your terminal to extract the file in the proper directory**
```
$ mkdir celeba
$ unzip </path/to/img_align_celeba.zip> -d </path/to/celeba>
```
The resulting directory structure should be:
```
/path/to/celeba
    -> img_align_celeba
        -> 188242.jpg
        -> 173822.jpg
        -> 284702.jpg
        -> 537394.jpg
           ...
```
<br>

**Note**: You may use any other dataset of your choice. However, please ensure that the directory structure remains the same for the code to be compatible with it. 
