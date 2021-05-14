*This task is taken originally from the Machine Learning Course (Spring 2020) lectured by Prof. Hung-yi Lee at National Taiwan University.*\
*For more information about the lecture on Adversarial Attack, please watch it on [YouTube](https://www.youtube.com/watch?v=NI6yb0WgMBM&ab_channel=Hung-yiLee).*

# Adversarial Attack
Adversarial machine learning is a machine learning technique that attempts to fool models by supplying deceptive input. The most common reason is to cause a malfunction in a machine learning model.\
Most machine learning techniques were designed to work on specific problem sets in which the training and testing data are generated from the same statistical distribution (IID). When those models are applied to the real world, adversaries may supply data that violates that statistical assumption. This data may be arranged to exploit specific vulnerabilities and compromise the results. ([from WikiPedia](https://en.wikipedia.org/wiki/Adversarial_machine_learning))

# Task Description
## Fast Gradient Sign Method (FGSM)
<p align="center"><img width="35%" src="https://i.imgur.com/mLopBSS.jpg"></p>
The equation for generating the perturbed adversarial image is shown here.

## To Do
In this tutorial, we are going to choose a proxy network to attack the block box by the FGSM method.

## Evaluation Metrics
- *Average L-inf. norm* between all input images and adversarial images
- *Success rate* of your attack
- Priority: Success rate > Ave. L-inf. norm

# Dataset
The dataset contains three files.
1. 200 RGB images with the size of (224 x 224). For example, 000.png ~ 199.png.
2. `categories.csv`: 1000 categories (0 ~ 999).
3. `label.csv`: information on each image.

# Reference
- [Machine Learning Course (Spring 2020) by Prof. Hung-yi Lee at National Taiwan University](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)
