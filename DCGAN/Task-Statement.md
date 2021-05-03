*This task is taken originally from the Machine Learning Course (Spring 2020) lectured by Prof. Hung-yi Lee at National Taiwan University.*\
*For more information about the lecture on GAN, please watch it on [YouTube](https://www.youtube.com/watch?v=DQNNMiAP5lw&list=PLJV_el3uVTsMq6JEFPW35BCiOQTsoqwNw&ab_channel=Hung-yiLee).*
# DCGAN (Deep Convolutional GAN)
Till now, many GAN structures are being proposed. Generally speaking, the architecture of GAN contains a Generator and a Discriminator.\
The one we will implement in this tutorial is the easiest one called *DCGAN*. The architecture of DCGAN is proposed by [Goodfellow et al](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf). in 2014.
## Task Description
In the tutorial, we will leverage the DCGAN model to produce several anime faces by Generator. The goal is to generate the anime faces from random noise and make them as real as possible after the *Generator*.
## Dataset
The dataset for training the DCGAN model was generated from [Crypko.ai](https://crypko.ai/#). However, they do not provide the function of anime image generation anymore. Hence, the dataset was formerly collected. (You can download it from the Google Drive link in the tutorial.)
## Result
The model achieves 0.0962 and 3.9073 for Discriminator and Generator loss, respectively.\
We also generate an image from random noise. The result is as follows.
<p align="center"><img width="90%" src="https://github.com/starkbao/Machine-Learning-Tutorial/blob/main/DCGAN/DCGAN_Result.png" /></p>
