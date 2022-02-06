# DL_CV
## Practiced basic Deeplearning skills, especially in field of Computer Vision, mainly using Keras

### Conv_Practice_01
+ Basic practice of using Conv2D layer, Pooling
+ Creating basic CNN Model
+ Preprocessing Fashion MNIST data and model training
+ O = (I - F + 2P)/2 + 1 Formula

### CIFAR10_Custom
+ Visualisation of CIFAR10 dataset
+ Data preprocessing
+ Recognised the difference of 'sparse_categorical_crossentropy' and 'categorical_crossentropy'
+ Tried He Normal instead of the basic weight initialisation of Keras, glorot_uniform
+ Tried Batch Normalisation and shuffle

### CIFAR10_Custom_CB_GAP_WR
+ Tried to use some Callbacks; ReduceLROnPlateau for the dynamic change in Learning rate and EarlyStopping to avoid overfitting
+ Tried Global Average Pooling, instead of Flatten. (A stage of Flatten layer -> Classification Dense Layer requires a lot of parameters, and it increases a chance of overfitting and increase of training time)
+ Tried l1, l2, l1_l2 weight regularisations per each layer by using tensorflow.keras.regularizers

### ImageDataGenerator_Aug
+ Practiced some augmentations with ImageDataGenerator
+ Including H,V Flips, Rotation, Shift, Zoom, Shear, Bright, Channel Shift, ZCA Whiteing, Normalisation

### CIFAR10_Custom_Aug
+ Included data augmentations in the previous CIFAR10_Custom to check whether there is any difference in a performance metric

### CIFAR10_Pretrained 
+ Practiced a way of using pretrained model in Keras Framework. 

### CatnDog_Gen
+ Downloaded cat-and-dog dataset from a kaggle dataset
+ Practiced how to read the directory and jpeg file names to create their absolute directory and made them as a dataframe
+ Practiced flow_from_directory(), flow_from_dataframe()

### Albumentations_Aug
+ Practiced how to use albumentation to created an augmented image.
+ Including Flip, Rotation, ShiftScaleRotation, Compose, Crop, RandomBrightnessContrast, HueSaturationValue, RGBShift, ChannelShuffle, ColorJitter, Gaussian Noise, Cutout, CoarseDropout, CLAHE, Blur, GaussianBlur

### CatnDog_Sequence
+ Created a Dataset instance that inherits Keras Sequence

### AlexNet_Practice
+ Created the layer structure of AlexNet 

### VGG_Practice
+ Created the layer structure of VGG16
+ Created a function conv_block() which creates the consecutive Conv layers as a block.

### Inception_Practice
+ Created the layer structure of GoogLeNet
+ Created a function inception_module to create the characteristical 'Inception module', including 1x1 Convolution.

### ResNet_Practice
+ Created the layer structure of ResNet
+ Created a function identity_block() to create the characteristical 'Identity block', including shortcuts

### CatnDog_Fine_Tuning
+ Learned the 'trainable' attribute of layers
+ Learned how to freeze the feature extractor parts of a pretrained model and how to unfreeze it after a certain number of epochs

### Learning_Rate_Scheduler
+ Learned how to build a scheduler function which will be inserted into the LearningRateScheduler callback object
+ Learned Step Decay, Cosine Decay, Cosine Annealing, and Ramp Up and Step Down Decay

