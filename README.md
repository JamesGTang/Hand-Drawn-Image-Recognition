> Written with [StackEdit](https://stackedit.io/).
## Project Description 
Abstract—’Quick Draw’ doodles dataset is a large hand drawn image dataset provided by Google. In this project, the subset of the ’Quick Draw’ doodles dataset is analyzed. The goal of this project is to classify the hand drawn doodles into 31 classes. In this report, we show the outstanding performance of CNN against other models. We also show the careful preprocessing on images, proper features selection, and the model construction significantly improve the performance of the models.

This project is for Kaggle Competition of COMP 551 (Applied Machine Learning) at McGill University. 

#### For full report please refer to the the report.pdf

## Introduction
Image analysis is a popular area of research in the domain of Machine Learning. Tackling a variant of the Google’s ’Quick Draw!’ dataset, the goal is to classify hand drawn images into 31 classes. The provided training and test datasets contain 10,000 labelled and unlabelled hand-drawn images respectively. Each image of size (100,100) consists of a doodle surrounded by randomly generated noise, where for one class labelled empty, each image consists of no doodle and only noise. In order to maximize prediction accuracy, different machine learning algorithm methods from baseline to more complex models are tested, including SVM1, neural network, k-NN, and CNNs. We use OpenCV for image preprocessing, and algorithms are built using scikit-learn and tflearn.

## Feature Design
### Thresholding and Denoising
Given the nature of the dataset, the first step in preprocessing entails removing noise from each image without altering the original doodle. Starting with image thresholding, we compare pixel values of the input image f with some threshold T and make a binary decision to output a binary image g.

![enter image description here](https://raw.githubusercontent.com/JamesGTang/Hand-Drawn-Image-Recognition/master/graph/original.png?token=AWCtz5x8MmAlX9LVn2vdHLQ-gusx1wzKks5cNEqLwA==)

Since the doodle and the noise share similar pixel values, binarizing the image according to a chosen threshold can either reduce noise ever so slightly or alter the drawing of interest. Hence, this step is intended to sharpen the image, as a stepping stone before extracting the largest connected component.

Using the OpenCV function connectedComponentsWith- Stats(), the image is broken into a series of connected components with 8-way connectivity (i.e. pixels are grouped as a component if any of the eight neighboring pixels connect). Given the ability to check for size, all components except the largest component is interpreted as noise and thus removed, isolating the supposed doodle.

![enter image description here](https://raw.githubusercontent.com/JamesGTang/Hand-Drawn-Image-Recognition/master/graph/denoise.png?token=AWCtz7473R_zWLfnXm7h5yBpW6URr11oks5cNEriwA==)
### Cropping 
The mentioned OpenCV function provides the starting coordinates and size of the bounding box surrounding the largest connected component. Since the bounding box is a rectangle with differing width/height, the greatest dimension is taken to
be both width and height of a square image. Before finalizing the mask to crop this bounding box of interest, the image is further padded slightly. Once cropped, all images are resized to (35,35) and binarized once more as resizing introduces nonbinary pixel values.
### Augmentation
The final step in the pre-processing is to augment the image to add variances of each image as a supplement to the 10,000 images dataset provided. Image augmentation provides the model with more data to train. For our image augmentation, we generate randomly rotated images, mirrored images, randomly shifted images and for each of the rotated image, we randomly generate a piecewise-affine variation (locally distorted image) and a general-affine variation (globally distorted image). This step is only performed before carrying out the CNN algorithm.

![enter image description here](https://raw.githubusercontent.com/JamesGTang/Hand-Drawn-Image-Recognition/master/graph/image_aug.jpg?token=AWCtz9rt63Vek9oQ1P6ckn_pQikmhOQFks5cNEtWwA==)

## CNN
Our final model is Convolution Neural Network, which is one of the most popular models in computer vision for image classification. Different from Feed Forward Neural Networks, CNN model feeds in the input to the next layer in small patches, such that their local connectivity allow better feature detections, that include edges, corners. Some of those features are hard to detect in fully connected neural networks. We choose to introduce multiple hidden layers within the network. Hence, our CNN model consists of three convolutional layers, all of which are followed by average-pooling layers which perform a down sampling operation to reduce the variance of the model to reduce overfitting, and also acting as a filter to the image for better classification. [4] Finally two fullyconnected layers with a final 31-way softmax are followed. We use a very efficient GPU implementation of the convolution operation to make training faster. To reduce overfitting in the fully-connected layers we set up the dropout rate and L2 regularization that proved to be very effective.

![enter image description here](https://github.com/JamesGTang/Hand-Drawn-Image-Recognition/raw/master/graph/tensorgraph.png)

## Methodology
### Training Pipeline
Using the proper loss function, in general we train our classifier as follows: 
 1. Perform the same preprocessing on both training and testing dataset as described in the feature design section.
 2. Transform the text labels of training set to one-hot encoding. 
 3. Split our 10,000 training images into training, validation and test set with ratio 8:1:1. For the Feed-forward Neural Network, other ratio splits are compared. 
 4. Reshape all samples to proper input size. 
 5. Use the training set to train the model and tune hyper-parameters base on the accuracy of the validation set. As cross validation is implemented for the Feed-forward Neural Network, tune hyper-parameters on the average accuracy of the validation set across all folds. 
 6. Evaluate the model base on the performance on test set. 
 7. Repeat step 4 and 5 to improve the model. 8) Use the best model to predict the final test set and make the submission.
### Hyper-parameter Tuning & Regularization
CNN: For augmentation part, we first use tensorflow do images augmentation by adding random vertical and horizontal flips, and random rotations with max 30 degrees of angle. For the tunning part, in order to maximize accuracy on test set, we use the CNN model with several hyper-parameters that include epoch number, learning rate, dropout rate, number of filters, kernel sizes and stride sizes. The corresponding range is listed below. Besides these hyper parameters, we also tries batch normalization after each convnet layer sets and different number of layers as the original models to tune those parameters. Then we use the validation set accuracy as a performance measure, we select the best parameters using grid search on all the hyper-parameters proposed.
![enter image description here](https://github.com/JamesGTang/Hand-Drawn-Image-Recognition/raw/master/graph/hyperparam.png)
## Results
### Performance
Among all four models we test, CNN model has the best performance without tuning with a validation accuracy of 73%. Upon the addition of several layers and fine tunning hyper parameters, we successfully obtain an accuracy of 78.4%. We finally choose our CNN model with number of filters[64, 128, 256], kernel size [2,2] for first layer and [3,3] for all other layers . For the epoch number, as we can observe in the graphs, after epoch 80, the accuracy does not increase but goes down. This indicates training more epochs would lead increasing on training accuracy but overfitting on validation accuracy. Besides, it is found that 60 epochs gave the best accuracy. 
![enter image description here](https://github.com/JamesGTang/Hand-Drawn-Image-Recognition/raw/master/graph/acc.png)
The Dropout Rate does not seem that sensitive to the accuracy when it is greater than 0.3. With fine-tuning the dropout rate from 0.1 to 0.9, and using the learning rate with 1e-3 and 60 epochs, the dropout of 0.5 is found to best regularize a model. However, different learning rates do affect the speed for growth accuracy a lot. We choose several values for the learning rate that varies from 1e-1 to 5e-4 and keep the other parameters unchanged(60 epochs, 0.5 dropout rate). Then it turns out that the learning rate with 5e-4 yields the best accuracy. Finally, for the batch normalization, we add it after each convnet layers, but this turns out do not significantly impact the validation accuracy.

![enter image description here](https://github.com/JamesGTang/Hand-Drawn-Image-Recognition/raw/master/graph/result.png)
### Metrics
![enter image description here](https://github.com/JamesGTang/Hand-Drawn-Image-Recognition/raw/master/graph/metrics.png)
### Submitted CNN Model Summary

## Conclusion 
Among all the classifiers, CNN model has the best performance with best accuracy 0.784 for test set. Different from the NN model, the neurons in CNN are arranged in 3 dimensions that are width, height and depth. And each layer transforms an input 3D volume to an output 3D volume with some differentiable functions that may or may not have parameters. These features of CNN allow it to take advantage of the two dimensional structure of the input data, and also allows the CNN to recognize specific shapes in the images, such as circle and curve. This special characteristic of CNN allows it to perform better for image classification problems comparing to other neural networks and baseline classifiers. 

![enter image description here](https://github.com/JamesGTang/Hand-Drawn-Image-Recognition/raw/master/graph/cnn-01.jpg)

The weakness of CNNs lays in the amount of data we have. CNNs have a large number of parameters, and for our image classification problem, we only have a small training set to work with. With the small data-set, we often over-fits the model during training. Since we only have 10,000 training data but 31 classes in total, CNN did not perform as good as we expected. 

![enter image description here](https://github.com/JamesGTang/Hand-Drawn-Image-Recognition/raw/master/graph/penguin.png)

Also for the specific classes such as penguin shown in the figures above, after cropping, it becomes almost unidentifiable because all the nearby pixels become connected. This makes it difficult for our CNN model to identify patterns from this class. From figure 13 below, we observe that almost all pixelated classes have much lower f1-score and precision, like class ’squiggle’, ’penguin’ and ’spoon’.

## Areas of further work
Further work can be done to improve the accuracy. For the preprocessing part, the images such as squiggles can be improved by recognizing some specific patterns. Also, the images’ distinguishability can be improved, the lines and patterns inside the images can be recognized better thus the accuracy could also be improved. To avoid losing the important part in the images, there are also some data augmentation techniques such as ’random crop’, ’random denoise’ or ’add gaussion noise’ can be used for further experiment. Like discussed above, all of the models can benefit from adding sharpening to the preprocessing as well. 

For the SVM model, we suggest to explore the nonlinear kernel trick to increase the nonlinearity of the model to better accommodate the image classification problem. Since Neural networks are expected to be less sensitive to noise than statistical regression models, perhaps testing the network with less preprocessing may result in better accuracy. Furthermore, as the impact of the number of hidden layers and choice of activation functions can be more thoroughly explored, there is room for improvement. 

For the CNN model, we can try add batch size as another hyper parameters to train. Since training neural networks in batch helps reduce memory and reduce training time. We can also try a deep CNN by adding more layers to the model. Furthermore, it is worthwhile to perform an extended hyperparameter tuning on the CNN model. Since CNN has many hyper-parameters like filter size and kernel size for convolution layers, stride for pooling layers and number of neurons for each fully connected layers. It is difficult to find the best set of hyper-parameters for our model given the time-frame of this project. But it is indeed likely to find a better set of hyperparameters if we extend our range of selection. And the CNN model can also be improved by implementing a ”sharpen’ procedure into our preprocessing process where each cropped drawing undergoes a filter that sharpens the image and reduce pixel density without changing the image. Unsurprisingly. this filter can be implemented using CNN as well, because of CNN’s ability solve image-related machine learning problems. Alternatively, we can perform transfer learning with the preexisting image CNN models and tweak the model in some way such that it fits the need of our task. 

Another technique that is worth to test out is ensemble modeling. That is, the dataset is trained on different models and then use boosting or bagging to combine the result from each model to give an overall prediction on the input data. Ensemble modeling also prevents the model from overfitting.

## Credit
### Team Bernoudles
He, Jingyi 
Rahman, Aanika
Zhao, Zhuoran
Tang, James
