*This task is taken originally from the Machine Learning Course (Spring 2020) lectured by Prof. Hung-yi Lee at National Taiwan University.*\
*For more information about the lecture on Life-long Learning, please watch it on [YouTube](https://www.youtube.com/watch?v=7qT5P9KJnWo&feature=youtu.be&ab_channel=Hung-yiLee)*.

# Introduction: Life-long Learning
- The approaches of life-long learning can be categorized into three main categories according to this [dissertation](https://arxiv.org/pdf/1910.02718.pdf).
- In the research, the author investigated the models proposed from 2016 to 2019 and can be summarized as follows:
  - Replay-based methods
  - Regularization-based methods
  - Parameter isolation methods

<p align="center"><img width="70%" src="https://i.ibb.co/VDFJkWG/2019-12-29-17-25.png" /></p>

- In this tutorial, we will implement two *prior-focused* methods from *regularization-based methods*. That is, the ***EWC*** and ***MAS***.


# Task Description
In this tutorial, three datasets are representing three tasks to be learned by a single model. The datasets are MNIST, SVHN, and USPS datasets.\
Besides the baseline model, the EWC and MAS algorithm are the ones we will investigate the model performance.

# Dataset
The dataset can be downloaded from PyTorch API, which you will find in the `.ipynb` file.

# Result
As we can see from the results below, thanks to the EWC and MAS, the accuracy on the validation set won't decrease drastically once the model continues learning new tasks.
<p align="center"><img width="40%" src="https://i.imgur.com/mCmuJZz.jpg" /></p>


# Reference
- [Machine Learning Course (Spring 2020) by Prof. Hung-yi Lee at National Taiwan University](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)
- [Continual Learning in Neural Networks](https://arxiv.org/pdf/1910.02718.pdf)
