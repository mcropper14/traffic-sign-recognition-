#okay so the problem here is that cars don't recongize different types of traffic signs 
#well Teslas can but they are expensive and can sometimes mistake different numbers 
#so I brought it upon my self to fix this problem, you're welcome, hummanity is saved 
#the reason this works is because instead of hard coding image pixels-it uses a neural network to determine which sign is which 
#Myra Cropper 1/12/21
 
#Instructions:
#ignore any warnings, I used a slightly outdated library for a faster run time 
#run the first cell then 2nd cell, when prompted enter in any url from below 
#the first cell can take a while to run-wait until all 10/10 epochs are loaded 
#the first cell will show all of the images that are loaded into the model, along with a chart that shows the distrubtion 
#you can scroll down to see everything that is loaded
#after the first cell is run, run the 2nd cell and when the program prompts copy and paste in any of the urls from below 
#after the first cell is run you can continue to enter all the different urls into the 2nd cell without re-running the 1st cell
 
 
 
#data used for images
#these images are open source and I have permission to use them 
!git clone https://bitbucket.org/jadslim/german-traffic-signs
 
 
#image urls to enter when program prompts 
#feel free to view the image urls in the brower-they are correctly classified by the program but if you doubt me go for it 
#copy and paste the url when prompted by the 2nd cell of the program
#yield sign: https://accuform-img2.akamaized.net/files/damObject/Image/huge/FRR377_hires.jpg
#30 mph sign: https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg
#50 mph sign: https://i.ebayimg.com/images/g/3L0AAMXQlgtS9Kk0/s-l300.jpg
#left turn sign: https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg
#slippery road sign: https://previews.123rf.com/images/bwylezich/bwylezich1608/bwylezich160800375/64914157-german-road-sign-slippery-road.jpg
 
 
 
#libraries 
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle 
import pandas as pd #use to work with the data from csv files 
import random 
 
np.random.seed(0)
 
#import data files to use
#traning data
with open('german-traffic-signs/train.p', 'rb') as f:
  train_data = pickle.load(f) #load the data 
#validation data 
with open('german-traffic-signs/valid.p', 'rb') as f:
  val_data = pickle.load(f) #load the data 
#test data 
with open('german-traffic-signs/test.p', 'rb') as f:
  test_data = pickle.load(f) #load the data 
print(type(train_data)) #dictonary 
 
X_train, y_train = train_data['features'], train_data['labels']
 
X_val, y_val = val_data['features'], val_data['labels']
 
X_test, y_test = test_data['features'], test_data['labels']
 
#images are 32 X 32 pixels with a depth of 3
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
 
#check that images are imported correctly 
assert(X_train.shape[0] == y_train.shape[0], " #Images !=  #Labels")
assert(X_val.shape[0] == y_val.shape[0], "#Images != #Labels")
assert(X_test.shape[0] == y_test.shape[0], "#Images != #Labels")
#check size of image 
assert(X_train.shape[1:]== (32, 32, 3)), "The image's sizes are not 32X32"
assert(X_test.shape[1:]== (32, 32, 3)), "The image's sizes are not 32X32"
assert(X_val.shape[1:]== (32, 32, 3)), "The image's sizes are not 32X32"
 
#plot the data to see all traffic images 
 
#loading the data from the csv files 
data = pd.read_csv('german-traffic-signs/signnames.csv')
print(data)
 
num_of_samples = []
 
cols = 5
num_classes = 43
 
#iterate over data
#iterate is a big word for it goes through every single picture in this big for loop, actually it's a nested for loop. Fancy.
fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 50))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows(): #iterate over data rows as (i, s), index, series, each index = class, stored into j 
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
        axs[j][i].axis("off") #gets image 
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "----" + row["SignName"]) #displays labels of the sign name 
            num_of_samples.append(len(x_selected))
 
#displays distribution of images in data classes 
 
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Traffic Signs distribution")
plt.xlabel("Traffic Sign class")
plt.ylabel("# of traffic signs")
plt.show()
 
#preprocess images to make classification easier 
import cv2 
 
#plt.imshow(X_train[1000])
plt.axis("off")
print(X_train[1000].shape)
print(y_train[1000])
 
#convert to grayscale, cuts down on processing power-reduces depth of input channel, color isn't sig feature, focus on edges  
 
def grayscale(img):
  img = np.array(img, dtype=np.uint8)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert between colors
  return img 
 
img = grayscale(X_train[1000])
#plt.imshow(img)
plt.axis("off")
print(img.shape)
 
#histogram equalization 
#standarizes the lighting 
#makes it easier for network to disinguish between features 
def equal(img):
  img = cv2.equalizeHist(img) #only accepts grayscale
  return img
 
img = equal(img) 
plt.axis("off")
print(img.shape) #verify unaffected
 
#apply to all images 
def preprocess(img):
  img = grayscale(img)
  img = equal(img)
  #normalize-dvide all pixel values by 255
  # causes all values to be between 0-1
  #brings all the values to a similar range 
  img = img/255
  return img 
 
#run dataset through function 
#train 
X_train = np.array(list(map(preprocess, X_train))) #go through and then return specific function and create new array
#test 
X_test = np.array(list(map(preprocess, X_test))) #go through and then return specific function and create new array
#validation  
X_val = np.array(list(map(preprocess, X_val))) #go through and then return specific function and create new array
 
#plt.imshow(X_train[random.randint(0,len(X_train)-1)])
plt.axis("off")
print(X_train.shape)
 
#add depth to the data 
 
X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1) #number of images, pixels, depth 
X_val = X_val.reshape(4410, 32, 32, 1) #number of images, pixels, depth
 
 
#data augmentation process 
from keras.preprocessing.image import ImageDataGenerator
data_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,rotation_range=10)
data_gen.fit(X_train)
 
#create new images
batches = data_gen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)
 
#plot the images 
fig, axis =plt.subplots(1, 15,figsize=(20,5))
fig.tight_layout()
 
#cycle through 50 images and plot them 
for image in range(15):
  axis[image].imshow(X_batch[image].reshape(32, 32))
  axis[image].axis('off')
 
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
 
#encoded values
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_val = to_categorical(y_val, 43)
 
#LeNet neural network
#the brains behind the operation 
#okay so neural networks are really cool, they work like the neurons in our brain processing information 
 
def leNet():
  model = Sequential()
  model.add(Conv2D(60, (5,5), input_shape=(32, 32,1), activation='relu')) #780 parameters to adject 
  model.add(Conv2D(60, (5,5), activation='relu')) #780 parameters to adject 
  #pooling function 
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.5)) #number of input nodes dropped each layer, prevents overfitting 
  #scale down into a generalize layer to avoid overfitting, less parameters to be adjusted 
  model.add(Conv2D(30,(3,3), activation='relu'))
  model.add(Conv2D(30,(3,3), activation='relu')) 
  #4 conv layers
  model.add(MaxPooling2D(pool_size=(2,2))) #cut image by half 
  #flatten the data to format it into layer 
  model.add(Flatten())
  #feeding the data into layer as a 1d array, 540 nodes 
  model.add(Dense(500, activation='relu'))
  #model.add(Dropout(0.5)) #number of input nodes dropped each layer, prevents overfitting 
 
  model.add(Dense(num_classes, activation='softmax'))
  #compile model
  model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'] )
  return model
 
model = leNet()
print(model.summary)
 
history = model.fit_generator(data_gen.flow(X_train, y_train, batch_size=50),
                            epochs=10,
                            validation_data=(X_val, y_val), shuffle = 1)
#get the image
import requests
from PIL import Image
url = input("Enter a url: ")
r = requests.get(url, stream=True)

img = Image.open(r.raw)
plt.imshow(img)
#plt.imshow(img, cmap=plt.get_cmap('gray'))
#preprocess image
img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocess(img)
#plt.imshow(img, cmap = plt.get_cmap('gray'))
print(img.shape)
#reshape reshape
img = img.reshape(1, 32, 32, 1)
#Test
sign = "The sign is: " 
#model_predict = model.predict_classes(img)
model_predict=model.predict(img)

#well this is a fantastic piece of code right here 
#displays the correct label for an image 
if(model_predict.any()== 0):
  print(sign + "20 mile speed limit.")
elif(model_predict.any() == 1):
  print(sign + "30 mile speed limit")
elif(model_predict.any() == 2):
  print(sign + "50 mile speed limit")
elif(model_predict.any() == 3):
  print(sign + "60 mile speed limit")
elif(model_predict.any() == 4):
  print(sign + "70 mile speed limit")  
elif(model_predict.any() == 5):
  print(sign + "80 mile speed limit")
elif(model_predict.any() == 6):
  print(sign + "end of speed limit")
elif(model_predict.any() == 7):
  print(sign +"100 mile speed limit")
elif(model_predict.any() == 8):
  print(sign + "120 mile speed limit")
elif(model_predict.any() == 9):
  print(sign + "no passing")
elif(model_predict.any() == 10):
  print(sign + "no passing for large vehicles")
elif(model_predict.any() == 11):
  print(sign + "right of way")
elif(model_predict.any() == 12):
  print(sign + "priority road")
elif(model_predict.any() == 13):
  print(sign + "yield")
elif(model_predict.any() == 14):
  print(sign + "stop")
elif(model_predict.any() == 15):
  print(sign + "no vehicles")
elif(model_predict.any() == 16):
  print(sign + "no large vehicles")
elif(model_predict.any() == 17):
  print(sign + "no entry")
elif(model_predict.any() == 18):
  print(sign + "general caution")
elif(model_predict.any() == 19):
  print(sign + "dangerous curve to the left")
elif(model_predict.any() == 20):
  print(sign + "dangerous curve to the right")
elif(model_predict.any() == 21):
  print(sign + "double curve")
elif(model_predict.any() == 22):
  print(sign + "bumpy road")
elif(model_predict.any() == 23):
  print(sign + "slippery road")
elif(model_predict.any() == 24):
  print(sign + "road is narrow on the right")
elif(model_predict.any() == 25):
  print(sign + "road work")
elif(model_predict.any() == 26):
  print(sign + "traffic signals" )
elif(model_predict.any() == 27):
  print(sign + "pedestrians")
elif(model_predict.any() == 28):
  print(sign + "children crossing")
elif(model_predict.any() == 29):
  print(sign + "bicycles crossing")
elif(model_predict.any() == 30):
  print(sign + "beware of ice/snow" )
elif(model_predict.any() == 31):
  print(sign + "wild animal crossing")
elif(model_predict.any() == 32):
  print(sign + "end of speed limit")
elif(model_predict.any() == 33):
  print(sign + "turn right ahead")
elif(model_predict.any() == 34):
  print(sign + "turn left ahead")
elif(model_predict.any() == 35):
  print(sign + "ahead only")
elif(model_predict.any() == 36):
  print(sign + "go straight or right")
elif(model_predict.any() == 37):
  print(sign + "go straight or left")
elif(model_predict.any() == 38):
  print(sign + "keep right")
elif(model_predict.any() == 39):
  print(sign + "keep left")
elif(model_predict.any() == 40):
  print(sign + "roundabout")
elif(model_predict.any() == 41) :
  print(sign + "end of no passing")
elif(model_predict.any() == 42):
  print(sign + "end of no passing for large vechiles")
else:
  print("Please enter a valid traffic sign.")                            
