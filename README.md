# PhysenNet

Tensorflow implementation of paper: [Phase imaging with an untrained neural network.](https://www.nature.com/articles/s41377-020-0302-3) We provide the experiment data for a quick demo.

## Citation
Fei Wang, Yaoming Bian, Haichao Wang, Meng Lyu, Giancarlo Pedrini, Wolfgang Osten, George Barbastathis and Guohai Situ. Phase imaging with an untrained neural network. *Light Sci Appl* **9**, 77 (2020).

## Abstract
Most of the neural networks proposed so far for computational imaging (CI) in optics employ a supervised training strategy, and thus need a large training set to optimize their weights and biases. Setting aside the requirements of environmental and system stability during many hours of data acquisition, in many practical applications, it is unlikely to be possible to obtain sufficient numbers of ground-truth images for training. Here, we propose to overcome this limitation by incorporating into a conventional deep neural network a complete physical model that represents the process of image formation. The most significant advantage of the resulting physics-enhanced deep neural network (PhysenNet) is that it can be used without training beforehand, thus eliminating the need for tens of thousands of labeled data. We take single-beam phase imaging as an example for demonstration. We experimentally show that one needs only to feed PhysenNet a single diffraction pattern of a phase object, and it can automatically optimize the network and eventually produce the object phase through the interplay between the neural network and the physical model. This opens up a new paradigm of neural network design, in which the concept of incorporating a physical model into a neural network can be generalized to solve many other CI problems.

## Pipeline
![avatar](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41377-020-0302-3/MediaObjects/41377_2020_302_Fig1_HTML.png?as=webp)

## How to use
**Step 1:** Configuring required packages
python 3.6

tensorflow 1.9.0

matplotlib 3.1.3

numpy 1.18.1

pillow 7.1.2

**Step 2:** Run main.py after download and extract the ZIP file.

## Results
![avatar](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41377-020-0302-3/MediaObjects/41377_2020_302_Fig5_HTML.png?as=webp)



