# DeepLearning
All things related to deep learning
train_test.py contains code to get the training.txt and testing.txt files.
deeplearning.py contains the code for classification of the images. The images used are from the Caltech 101 database.
Three SVM kernels, namely, linear, RBF and polynomial  were used to classify the data from the penultimate fully connected layer.
Best accuracy were obtained with the linear and polynomial kernels (Accuracy of 91%). The RBF kernel surprisingly failed to classify even a single image correctly :(
