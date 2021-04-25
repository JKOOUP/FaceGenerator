# Face generator: Deep Convolutional Generative Adversarial Network for face generation


### Introduction

Face generator is an application for face generation using DCGANs. Project developed in pyTorch for educational purposes. 

### Requirements 
Face generator uses python libraries listed in requirements.txt:
```
torch==1.8.0
numpy==1.20.1
torchvision==0.9.0
Pillow==8.1.2
gdown==3.12.2
```
### Installation
1. Install requirements:
```
pip install -r requirements.txt
```
2. Run bash script to install pre-trained generator and create some directories:
```
chmod +x setup.sh
./setup.sh
```
### Generation

To generate image use: 
```
python generate.py
```
In `./utils/config.py` you can define path to generator.

### Training
To train your own model use:
```
python train.py
```
You can specify training parameters in `./utils/config.py`.

### Dataset
For training you can use CelebA dataset with photos of celebrities:
```
gdown --id 15ieGCiDmjPvMMOXX0Evkb3OJBBrsQkeL
```
Also you can use your own dataset with simple structure:
```
./dataset/
	000001.jpg
	000002.jpg
	...
```
All parameters you can configure in `./utils/config.py`.

### Demo:
![all text](https://github.com/JKOOUP/FaceGenerator/tree/master/demo/FaceGenerator.png?raw=true)
