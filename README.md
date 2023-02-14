# Manga Text Segmentation using U-Net

In this project, I explored the problem of text segmentation in manga (Japanese comics). This work is heavily based on the work "A study on image segmentation with convolutional neural networks: text segmentation problem in manga" [(link)](https://bcc.ime.usp.br/tccs/2020/pedrohba/index.html) by Pedro Henrique Barbosa de Almeida. You can check the author's implementation [here](https://github.com/robonauta/MAC0499).

My implementation uses PyTorch and Tversky loss, while the original implementation uses TensorFlow 1.x and categorical cross-entropy. This project was undertaken purely out of personal interest and as an exercise, and I do not plan on actively maintaining this repository. Of course, the original work goes a lot deeper into the problem itself, as well as exploring different settings.

# Getting Started

* Clone the repository: `git clone https://github.com/andrekato1/unet-manga-text-segmentation.git`
* (Recommended) Create an Anaconda environment: `conda create --name manga-segmentation`
* Install PyTorch (CUDA): `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
* Install Albumentations: `conda install -c conda-forge albumentations`
* Install tqdm: `conda install tqdm`
* Run `python train.py` to train the model. Results are available inside the `saved_images` directory

# Dataset

The data used is the same as in the original work, which is the Manga109 dataset, containing volumes from 109 series published between 1970 and 2010. The target images were manually prepared by the original author.

# Model
The deep learning model used in this project is U-Net, a popular architecture for image segmentation, originally proposed to tackle the same segmentation problem, but for biomedical images.

# Results
You can find the results inside the `saved_images` folder, which contains segmented text from the images from the `data/TT_ds1` folder. For these particular results, the model was trained on the images from the `data/D1_ds1` folder.

As the author concluded, a model trained on a particular title can work quite well on a different series. Not only that, it only needed around 10 training images in order to get this result. For more details, please read the original work. 