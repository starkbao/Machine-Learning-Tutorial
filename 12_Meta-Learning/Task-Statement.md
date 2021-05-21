*This task is taken originally from the Machine Learning Course (Spring 2020) lectured by Prof. Hung-yi Lee at National Taiwan University.*\
*For more information about the lecture on Meta-Learning, please watch it on [YouTube](https://www.youtube.com/watch?v=EkAqYbpCYAc&ab_channel=Hung-yiLee)*.

# Introduction: Meta-Learning (MAML)
Meta-learning, also known as “learning to learn”, intends to design models that can learn new skills or adapt to new environments rapidly with a few training examples. There are three common approaches:
1. learn an efficient distance metric (metric-based);
2. use (recurrent) network with external or internal memory (model-based);
3. optimize the model parameters explicitly for fast learning (optimization-based).

We will not go through the detail of each approach. If you are interested in any of them, [this article](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html) might be helpful to you.

# Task Description
The task is reproduced based on a paper called [*Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (2017)*](https://arxiv.org/abs/1703.03400) by Finn et al. In the paper, they proposed a MAML algorithm for Few-Shot Supervised Learning, which can be categorized as an optimization-based approach. The detail of the algorithm can be found below.
<p align="center"><img width="60%" src="https://i.imgur.com/TRFd6AF.jpg" /></p>

In this tutorial, we are going to train a model on different tasks taken from the [Omniglot Dataset](https://github.com/brendenlake/omniglot). Then, test the model on different tasks other than training tasks.

# Dataset
The original Omniglot dataset can be found [here](https://github.com/brendenlake/omniglot). However, the dataset we will use is already preprocessed.

# Reference
- [Machine Learning Course (Spring 2020) by Prof. Hung-yi Lee at National Taiwan University](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)
- [Meta-Learning: Learning to Learn Fast](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)
- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (2017)](https://arxiv.org/abs/1703.03400)
