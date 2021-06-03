#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import imutils
import cv2
import io
import os
import random
import shutil
import matplotlib.pyplot as plt
import skimage 
from skimage import filters 
from skimage.feature import greycomatrix, greycoprops
from skimage import data


# In[2]:


# function for image processing with segmentation
def image_processing(image):
#     plt.imshow(image, cmap="gray")
#     plt.title('Input grayscale Images')
#     plt.xticks([]), plt.yticks([])
#     plt.show()
                #Gaussian Blur
    blurGaussian = cv2.GaussianBlur(image,(5,5),5)
                #mean blur
    kernel = np.ones((5,5),np.float32)/25
    meanblur = cv2.filter2D(blurGaussian,-1,kernel)
                #Median Blur
    median = cv2.medianBlur(meanblur,5)
          #segmentation of images  using threshhold segmentation
    th =165
    max_value=255
    ret, out1 = cv2.threshold(median,th,max_value,cv2.THRESH_BINARY)
        #contours
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(out1)
#     plt.tick_params(axis='both', which='both',top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)
#     plt.title('Cropped Image')
#     plt.show()
                    #segmenatation of images using historigram
    #histimg=plt.hist(median.ravel(),256,[0,256])
#     plt.show()
    return out1


# # Feature Extraction

# In[3]:


def featureImage(resized_img):
    iar = np.asarray(resized_img)
#     print(iar)
    iar.max()
    # 'contrast',‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’
    glcm=g = greycomatrix(iar, [1,2], [0, np.pi/2], levels=260,normed=True, symmetric=True)

    contrast = greycoprops(glcm, 'contrast')
    print("the contrast of the images is ",contrast)
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    print("the dissimilarity of the images is ",dissimilarity)
    homogeneity = greycoprops(glcm, 'homogeneity')
    print("the homogeneity of the images is ",homogeneity)
    energy = greycoprops(glcm, 'energy')
    print("the energy of the images is ",energy)
    correlation = greycoprops(glcm, 'correlation')
    print("the correlation of the images is ",energy)
    ASM = greycoprops(glcm, 'ASM')
    print("the ASM of the images is ",ASM)


#     plt.plot(contrast,label="contrast")
#     plt.plot(dissimilarity,label="dissimilarity")
#     plt.plot(homogeneity,label="homogeneity")
#     plt.plot(energy,label="energy")
#     plt.plot(correlation,label="correlation")
#     plt.plot(ASM,label="ASM")
#     plt.title("GSLM MATRIX")
#     plt.xlabel("OFFSITE")
#     plt.ylabel("FEATURES")
#     plt.legend()
#     plt.show()


# In[4]:


size_img=int(input("enter the re-sized images"))
IMG_DIR="G:\\brainMydataSet"
CATEGORIES = ["newno","newyes"]
X=[]
Y=[]
def create_data_set():
    for categories in CATEGORIES:
        path = os.path.join(IMG_DIR,categories)
        #this function is used to cencatenate the path with sub folder
        #class_num can classified categories and marked it as 0 and 1 form.
        class_num = CATEGORIES.index(categories) 
        for img in os.listdir(path):
             # load the image\n",
            img_array =cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img_array,(size_img,size_img),interpolation=cv2.INTER_CUBIC)
            processed_img=image_processing(resized_img)
            #feature Extraction using glsm matrix
#             feature_image = featureImage(resized_img)
            # normalize values
            normal_img = processed_img/255
            # convert image to numpy array and append it to X
            X.append(normal_img)
            # append a value of 1 to the target array if the image is in the folder named 'yes', otherwise append 0.
            #from classnum
            Y.append(class_num)
    x=np.array(X)
    y=np.array(Y)
    
    print(f'Number of examples is: {len(x)}')
#     print(f'X shape is: {x.shape}')
#     print(f'y shape is: {y.shape}')
    
    return x,y


# In[5]:


create_data_set()


# In[6]:


# Shuffle two lists with same order 
# Using zip() + * operator + shuffle() 
temp = list(zip(X,Y)) 
random.shuffle(temp) 
X,y= zip(*temp) 


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


def split_data(X, y, test_size=0.2):
       
    """
    Splits data into training, development and test sets.
    Arguments:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    Returns:
        X_train: A numpy array with shape = (#_train_examples, image_width, image_height, #_channels)
        y_train: A numpy array with shape = (#_train_examples, 1)
        X_val: A numpy array with shape = (#_val_examples, image_width, image_height, #_channels)
        y_val: A numpy array with shape = (#_val_examples, 1)
        X_test: A numpy array with shape = (#_test_examples, image_width, image_height, #_channels)
        y_test: A numpy array with shape = (#_test_examples, 1)
    """
    
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# In[9]:


X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, Y, test_size=0.3)


# In[10]:


print ("number of training examples = " + str(np.shape(X_train)[0]))
print ("number of development examples = " + str(np.shape(X_val)[0]))
print ("number of test examples = " + str(np.shape(X_test)[0]))
print ("X_train shape: " + str(np.shape(X_train)))
print ("Y_train shape: " + str(np.shape(y_train)))
print ("X_val (dev) shape: " + str(np.shape(X_val)))
print ("Y_val (dev) shape: " + str(np.shape(y_val)))
print ("X_test shape: " + str(np.shape(X_test)))
print ("Y_test shape: " + str(np.shape(y_test)))


# In[11]:


print(np.ndim(X_train))
print(np.ndim(X_test))
print(y_train)


# In[12]:


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


# In[13]:


def build_model(input_shape):
    """
    Arugments:
        input_shape: A tuple representing the shape of the input of the model. shape=(image_width, image_height, #_channels)
    Returns:
        model: A Model object.
    """
    # Define the input placeholder as a tensor with shape input_shape. 
    X_input = tf.keras.Input(input_shape)# shape=(?, 128, 128, 1)
    print(X_input.shape)
    
    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input) # shape=(?, 244, 244, 3)
    print(X.shape)
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X) # shape=(?, 238, 238, 32)
    print(X.shape)
    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool0')(X) # shape=(?, 59, 59, 32) 
    print(X.shape)
    # MAXPOOL
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X) # shape=(?, 238, 238, 32)
    print(X.shape)
    X = MaxPooling2D((2, 2), name='max_pool1')(X) # shape=(?, 14, 14, 32)
    print(X.shape)
    # FLATTEN X 
    X = Flatten()(X)# shape=(?, 6272)
    print(X.shape)
    # FULLYCONNECTED
    X = Dense(1, activation='sigmoid', name='fc')(X) # shape=(?, 1)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='BrainDetectionModel')
    
    return model


# In[14]:


input_shape = (size_img, size_img,1)


# In[15]:


model = build_model(input_shape)


# In[16]:


model.summary()


# In[20]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[21]:


X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_val = np.asarray(X_val)
y_val= np.asarray(y_val)


# In[19]:


X_train = np.reshape(X_train,(1622,size_img, size_img,1))
X_test = np.reshape(X_test,(348,size_img, size_img,1))
X_val = np.reshape(X_val,(348,size_img, size_img,1))


# In[22]:


print("Fit model on training data")
modelhistory = model.fit(x=X_train, y=y_train, batch_size=16, epochs=30, validation_data=(X_val, y_val))


# In[23]:


modelhistory.history


# In[24]:


# plot model performance
acc = modelhistory.history['accuracy']
val_acc = modelhistory.history['val_accuracy']
loss = modelhistory.history['loss']
val_loss = modelhistory.history['val_loss']
epochs_range = range(1, len(modelhistory.epoch) + 1)

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Set')
plt.plot(epochs_range, val_acc, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Set')
plt.plot(epochs_range, val_loss, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')

plt.tight_layout()
plt.show()


# # To save our CNN model
# 

# In[ ]:


#To save model ..
model.save('BrainTumorDetection1.model')


# In[25]:


model.save('braintumor.h1')


# # Model testing 

# In[27]:


filepath1="G:\\brainMydataSet\\yes\\Y27.JPG"
filepath2="G:\\brainMydataSet\\no\\13 no.jpg"


# In[ ]:





# In[28]:


import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

CATEGORIES = ["no","yes"]


def prepare(filepath):
    IMG_SIZE = 128  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#     plt.imshow(img_array, cmap="gray")
#     plt.title('Input grayscale Images')
#     plt.xticks([]), plt.yticks([])
#     plt.show()
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    blurGaussian = cv2.GaussianBlur(new_array,(5,5),5)
                #mean blur
    kernel = np.ones((5,5),np.float32)/25
    meanblur = cv2.filter2D(blurGaussian,-1,kernel)
                #Median Blur
    median = cv2.medianBlur(meanblur,5)
          #segmentation of images  using threshhold segmentation
    th =165
    max_value=255
    ret, out1 = cv2.threshold(median,th,max_value,cv2.THRESH_BINARY)
#     plt.imshow(out1)
#     plt.title('threshold')
#     plt.xticks([]), plt.yticks([])
#     plt.show()
    normal_img = out1/255
    return out1.reshape(1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("braintumor.h1")

prediction = model.predict([prepare(filepath1)])
print(prediction)
# will be a list in a list.
cate=(int(str(float(prediction)).split('.')[0]))
print(cate)
print(CATEGORIES[cate])


# In[ ]:




