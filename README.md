# Image-Classification

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YY2ddQ1vP2T4ntUav0-Cwe9lePYvh2uC?usp=drive_open#scrollTo=953edfb2)


## Project Statement:

The goal of the project is to classify 13 different [Iranian Cars](https://www.kaggle.com/datasets/usefashrfi/iran-used-cars-datasethttp:// "Iranian Cars").
using AlexNet Model, VGG16-TransferNet, VGG16-ShallowNet and VGG16-SVM.

![DALL·E 2023-11-12 18 05 44 - The goal of the project is to classify 13 different Iranian Cars](https://github.com/JalehFar/Image-Classification/assets/117992631/8535c03b-310e-4368-97e5-b5497cb730d5)

### AlexNet:

A CNN model that is trained from the scratch. The total number of trainable parameters : 28,092,765.
<p align="center">
<img width="704" alt="image" src="https://github.com/JalehFar/Image-Classification/assets/117992631/872c3109-47d8-4961-8626-87a5db8d3c55">
</p>

As you can see our model is in underfitting situation after 20 epochs. This is very likely that if I train the model for more epochs, I will get better results because the loss on the train set is decreasing. However, I decided to train all of the models just for 20 epochs in order to save time and also have a fair comparison between the models. It took about 2 hours to train. And after 20 epochs we have 0.2618 accuracy on training set and 0.2761 on test set.


### VGG16-TransferNet:

Only the parameters of‘block5_conv3’ layer are trained. In addition, we add some layers to the VGG16 model and train only the last part and also the layers that we have added. With this trick we only have to train 4,554,125 parameters out of around 17 million. Which is much less than the number of parameters of AlexNet model that we saw in the previous section.

<p align="center">
<img width="704" alt="image" src="https://github.com/JalehFar/Image-Classification/assets/117992631/7a176ab1-f90a-4ec0-8ce6-b4a7454aa7dd">
</p>

The performance of the train and the test is very similar that it is acceptable for this type of problem and behaves positively with this dataset. We note how with the callback function we did not do all the 20 epochs we had done in the previous problem but only 17, this because after 3 consecutive epochs the test set had no improvements and therefore the execution was terminated.

### VGG16-ShallowNet:

We use VGG16 features trained on ImageNet as input for a simple classifier based on Shallow NN. With shallow net trick, only the last layer will be trained. So, from 14,897,485 parameters, only 161,293 parameters will actually be trained and all the others will be fixed.
<p align="center">
<img width="704" alt="image" src="https://github.com/JalehFar/Image-Classification/assets/117992631/7afdb3ac-476a-4405-b5a0-47e3c531f204">
</p>

We can recognize a slight overfitting problem here where at the end of the training the loss on the training set was decreasing but the loss on the test set was decreasing. However, the overall accuracy is 0.68 on the test set.

### VGG16-SVM:

We use VGG16 features to train an SVM model. We flattened the features and then used SVM model as a classifier. Training the SVM model took only 17 minutes. But the accuracy was relatively lower than other models except AlexNet. (Accuracy: 0.619)


## Conclusion:

To summarize, four models have been used for the IranCar dataset classification problem. AlexNet was not able to converge to optimal solution after around two hours of training and 20 epochs and ended with 27% accuracy with Adam optimizer. While, using transfer learning I was able to build much better models. The second model VGG16 trained on ImageNet could reach the best performance among all the others which was 75%. This model reached the accuracy of AlexNet in the second epoch and after about 30 minutes. In The last model I used VGG16 features to train an SVM model and that model reached 62% accuracy very fast and it took only 17 minutes for caching and other operations.


