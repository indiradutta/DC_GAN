# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Network (DCGAN) implementation
This is an implementation of the research paper <a href = "https://arxiv.org/abs/1511.06434.pdf">"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"</a> written by Alec Radford, Luke Metz, Soumith Chintala.

## Inspiration
Unsupervised Learning with CNNs has not recieved it's due attention. In thier paper, Alec Radford, Luke Metz and Soumith Chintala have tried to *"bridge the gap between the success of CNNs for supervised learning and unsupervised learning"*. DCGAN, as introduced in this paper, is an architecture that demonstrates the strength of CNNs as a candidate for Unsupervised Learning.

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

## Model Components
The DCGAN Architecture has the following components:

- The Generator uses fractional-strided convolutions followed by batch normalisation and ReLU activation for all layers except for the last that uses tanh activation.
- The Discriminator uses strided convolutions followed by batch normalisation and LeakyReLU activation for all layers except for a single sigmoid output.
<img src="https://miro.medium.com/max/846/1*rdXKdyfNjorzP10ZA3yNmQ.png" >

## Implementation Details
- The given batch of images are first upsampled through the fractional-strided convolutions of the Generator.
- Batch Normalization is applied to boost the learning process.
- The normalized images are passed thorugh a bounded activation function (ReLU) that helps in a faster convergence of the model and allows it to cover the entire spatial extent of the images.
- The images enter the Discriminator after passing through a final tanh layer. The discriminator uses strided convolutions for downsampling the images following a set of similar actions as the Generator except that LeakyReLU is used.

## Installation and Usage Guidelines
To use the repo and generate your own images please follow the guidelines below


- Cloning the Repository: 

        git clone https://github.com/indiradutta/DC_GAN
        
- Entering the directory: 

        cd DC_GAN
        
- Setting up the Python Environment with dependencies:

        pip install -r requirements.txt

- Running the file for training:

        python3 train_gen.py
        
The train_gen.py file takes the path to the dataset as *"/content/dcgan/celeba"* dataset by default. Please initialize the DCGAN module with your desired dataset path and train as:

```python
dc_gan = DCGAN(data = <path/to/dataset>)
img_list, G_losses, D_losses = dc_gan.train(<path/to/save/model>)
```

Incase you have either no GPU (0) or more than 1 GPU on your machine, consider changing the ngpu parameter while initializing the DCGAN module with your desired dataset path and train as:


```python
dc_gan = DCGAN(data = <path/to/dataset>, ngpu = <number of GPUs available>)
img_list, G_losses, D_losses = dc_gan.train(<path/to/save/model>)
```

Check out the standalone demo notebook and run dc_gan <a href = 'https://colab.research.google.com/github/indiradutta/DC_GAN/blob/main/demo/dcgan_standalone_demo.ipynb'>here</a>.

**Note**: Is is advisable to use a GPU for training because training the DCGAN is computationally very expensive.

- Running the file for inference:

        python3 test.py
        
The test.py file takes the path to the pre-trained as *"model.pth"* by default. Please initialize the Deep_Conv_GAN with the desired path to the model and get thr inferences as:

```python
Deep_Conv_GAN.inference(Deep_Conv_GAN, set_weight_dir='model.pth' , set_gen_dir='<path/to/save/inferences>')
```
Check out the standalone demo notebook and run inferences <a href = 'https://colab.research.google.com/drive/1C2vlQ2vR2fYGxkkSqAQNCAWxS0Hp_NrD?usp=sharing'>here</a>.

<hr>

## Results from Implementation

- Plot to see how D and G’s losses changed during training

<img src = "results/losses.png">

- Batches of fake data from G

<img src = "results/result.png" height = 350px width = 350px> &nbsp; &nbsp; <img src = "results/result2.png" height = 350px width = 350px>

- Training progression of G


https://user-images.githubusercontent.com/66861243/117669333-67e99600-b1c4-11eb-885c-bcb5e4b7b299.mp4

Please check out the result <a href = "docs/documentation.md">documentation</a>. 
## Contributors

- <a href = "https://github.com/indiradutta">Indira Dutta</a>
- <a href = "https://github.com/srijarkoroy">Srijarko Roy</a>
