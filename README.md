# Image-Processing-ML
This project includes three parts. All work on a GPU machine.

## 1. Face classification

Use Olivetti Faces dataset to buils a face classifier based on CNN.


[Code](https://github.com/Siyuqqq/Image-Processing-ML/blob/master/FaceClassification_CNN/FaceClassification_CNN.ipynb)

![](https://github.com/Siyuqqq/Image-Processing-ML/blob/master/FaceClassification_CNN/olivettifaces.gif?raw=true)

Finally achieve a test accuracy of 0.998.

[reference](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)



## 2. Image Generator
Using GAN (Generative Adversarial Networks), generate simulated picture based on given condition. The G's(generator) training goal is to deceive the D(discriminators) to make D believe that the fake image is match to the condition.

[Code](https://github.com/Siyuqqq/Image-Processing-ML/blob/master/ImageGenerator/ImageGenerator_GPU.ipynb)

First generated picture with poor acc value

![](https://github.com/Siyuqqq/Image-Processing-ML/blob/master/ImageGenerator/images/facades/0_0.png)

picture generated after 22 epoch training

![](https://github.com/Siyuqqq/Image-Processing-ML/blob/master/ImageGenerator/images/facades/20_200.png)

picture generated after 100 epoch

![](https://github.com/Siyuqqq/Image-Processing-ML/blob/master/ImageGenerator/images/facades/99_0.png)


[reference paper](https://arxiv.org/abs/1611.07004)

data loader and basic construct of GAN is based on the follow link

[data loader](https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/data_loader.py)



## 3. Style Transfer
Build a style transfer based on the pre-trained model VGG16.

[Code](https://github.com/Siyuqqq/Image-Processing-ML/blob/master/StyleTrans/StyleTrans_GPU3.ipynb)

The input image

![](/StyleTrans/input.png)

The style image

![](/StyleTrans/style.png)

The output image

![](/StyleTrans/output.png)

[reference](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
