# MNISTLearner
A simple machine learning model that attempts to predict the digits in the MINST datasset, which is a large database of handwritten digits that is commonly used for training various image processing systems. It uses pixel voting to make predicitions, where white pixels vote for the number which has occured the most times when that specific pixel was white.

The code applies a binary threshold to each image in the training and testing sets. This is done using the `cv2.threshold` function, which applies a fixed-level thresholding to each element in an array (in this case, the array is an image). Otsu Thresholding is used to find the best threshold to binarize the image. This is done to simplify the images and reduce the amount of data the model needs to learn from.

**MNISTLearner class**: This class is a simple implementation of a machine learning model for the MNIST dataset.

The `fit` method is used to train the model. It initializes a 2D list where each element is a dictionary that maps a label (0-9) to a count. It then iterates over each pixel in each image in the training set. If the pixel value is 255 (white), it increments the count for the corresponding label in the dictionary at the current pixel position.

The `predict` method is used to make predictions on a set of images. It iterates over each image in the input set and appends the prediction for each image to a list of predictions

The `predict_single` method is used to make a prediction on a single image. It initializes a dictionary that maps a label (0-9) to a count (initialized to 0). It then iterates over each pixel in the image. If the pixel value is 255 (white), it increments the count for each label according to the trained model. It then returns the label with the maximum count.


This is a very basic model and only provides an accuracy of roughly 68%. This model serves as a good starting point for understanding how machine learning models work and how can they be applied for image classification problems. It's also a good example of how to implement a simple model from scratch without using any machine learning libraries.
