## An Imperceptible Anti-forensic Method, Models and Dataset 

**Perception matters: exploring imperceptible and transferable anti-forensics forGAN-generated fake face imagery detection**

<img 
src="https://github.com/enkiwang/Imperceptible-fake-face-antiforensic/blob/master/example.png/"
width=800>


This repository contains the code, models and dataset for the project "Perception matters: exploring imperceptible and transferable anti-forensics for GAN-generated fake face imagery detection". 

## Implementation
This code has been tested on Ubuntu 16.04 system, with following pre-requisites. 

### Pre-requisites

1. python >=3.6.10
2. PyTorch >=0.4.1
3. torchvision >=0.2.1
4. Pillow >=5.4.1


### Dataset
The face dataset we used is an image subset dataset downloaded from [here](https://github.com/NVlabs/stylegan). 

If you agree with the license in [here](https://github.com/NVlabs/stylegan/blob/master/LICENSE.txt), you might be permitted to download the downsampled image subset from [here](https://drive.google.com/file/d/1tudf3eFtlPtn5eX7BWW1IxjXtfIgreyH/view?usp=sharing). 
   
This face image subset consists of 40,000 real face images and 40,000 fake face images with image resolution as 128x128. For real or fake face images, the dataset splits are: 30,000 images are used for model training; 5,000 images for validation; and the rest 5,000 for test. 

After downloading the dataset, please unzip and put them in the data directory.  

 

### Models
The pretrained deep learning-based fake face forensic models can be downloaded [here](https://drive.google.com/file/d/1Me6i2xKHbMdh7I-JS7ZJ1QN-6e-SBeov/view?usp=sharing). After downloading pretrained models, please put them in the checkPoint directory. 


### Run attacks 
```bash
cd attacks
bash ./run_attack.sh
```

If you find our code useful in your research, please consider citing our work: 







 
