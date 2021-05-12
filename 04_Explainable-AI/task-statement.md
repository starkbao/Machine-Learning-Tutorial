*This task is taken originally from the Machine Learning Course (Spring 2020) lectured by Prof. Hung-yi Lee at National Taiwan University.*\
*For more information about the lecture on Explainable AI, please watch it on [YouTube](https://www.youtube.com/watch?v=lnjrn3bF9lA&feature=youtu.be&ab_channel=Hung-yiLee).*

# Explainable AI
Explainable AI (XAI) is artificial intelligence (AI) in which the results of the solution can be understood by humans. It contrasts with the concept of the "black box" in machine learning where even its designers cannot explain why an AI arrived at a specific decision. XAI may be an implementation of the social right to explanation. XAI is relevant even if there is no legal right or regulatory requirement—for example, XAI can improve the user experience of a product or service by helping end-users trust that the AI is making good decisions. This way XAI aims to explain what has been done, what is done right now, what will be done next, and unveil the information the actions are based on. These characteristics make it possible (i) to confirm existing knowledge (ii) to challenge existing knowledge and (iii) to generate new assumptions. ([from Wikipedia](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence))

# Task Description
In this tutorial, we will go through three tasks, the Saliency Map, the Filter Visualization, and the Lime. All these tasks are different approaches in the field of Explainable AI.
## Task 1: Saliency Map
- Compute the gradient of output category with respect to input image.
<p align="center"><img width="50%" src="https://i.imgur.com/ew3D9H8.jpg"></p>

## Task 2: Filter Visualization
- Use Gradient Ascent method to find the image that activates the selected filter the most and plot them (start from white noise).
<p align="center"><img width="50%" src="https://i.imgur.com/TUkfdwZ.jpg"></p>
<p align="center"><img width="50%" src="https://i.imgur.com/MoTuDwi.jpg"></p>

## Task 3: Lime
- Use the Lime method to crack a image to small interpretable segements.
<p align="center"><img width="50%" src="https://i.imgur.com/dSfNVYf.jpg"></p>
<p align="center"><img width="50%" src="https://i.imgur.com/VdQcBVn.jpg"></p>
<p align="center"><img width="50%" src="https://i.imgur.com/FWahVnz.jpg"></p>

- By this technique, we can know which part of the image the model is seeing and thus making the decision.
<p align="center"><img width="50%" src="https://i.imgur.com/iX2AKht.jpg"></p>



# Dataset
The food dataset in this tutorial is the same as 02_CNN.
The dataset file containing three folders, namely training, validation, and testing. The naming format of an image in the training and validation set is *class_number.jpg*. For example, *3_100.jpg* meaning that it's class 3. The naming format of a testing image is *number.jpg*.\
You can download the dataset in this [Google Drive link](https://drive.google.com/file/d/19CzXudqN58R3D-1G8KeFWk8UDQwlb8is/view).


# Reference
- [Machine Learning Course (Spring 2020) by Prof. Hung-yi Lee at National Taiwan University](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)
