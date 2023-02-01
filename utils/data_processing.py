import cv2
import uuid
import os
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import matplotlib as plt
from tensorflow.keras.models import Model

def capture_img(positive_path, anchor_path, negative_path):
    # Setup paths
    #os.path.join joins together different paths
    POS_PATH = os.path.join('data', positive_path)  #positive verification
    NEG_PATH = os.path.join('data', negative_path)  #negative verification
    ANC_PATH = os.path.join('data', anchor_path)
    #the anchor is your input and is to be seen if it matches your positive image. 
    cap = cv2.VideoCapture(0) #connection to the webcam
    while cap.isOpened(): #loop every single frame of our webcam
        ret, frame = cap.read()  #read the frame
   
        #    Cut down frame to 250x250px
        frame = frame[200:200+250,570:570+250, :]
    
        # Collect anchors 
        if cv2.waitKey(1) & 0XFF == ord('a'):
            # Create the unique file path 
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())) #use uuid1 to create an unique identifier
            # Write out anchor image
            cv2.imwrite(imgname, frame)
    
        # Collect positives
        if cv2.waitKey(1) & 0XFF == ord('p'):
            # Create the unique file path 
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out positive image
            cv2.imwrite(imgname, frame)
    
        # Show image back to screen
        cv2.imshow('Image Collection', frame)
    
        # Breaking gracefully
        if cv2.waitKey(1) & 0XFF == ord('q'): #wait 1 millisecond
           break
        
    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()
    return frame

#frame = capture_img()

anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(300)  
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(300)

dir_test = anchor.as_numpy_iterator()


def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img

img = preprocess('data/fanchor/17b69f48-1ee7-11ed-9efc-7adac6cf2095.jpg')
#print(img)
img.numpy().max() 
#dataset.map(preprocess)

#create a labelled dataset
#positive has tuples of anchor and positive, zipped so to iterate at once
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

samples = data.as_numpy_iterator()

exampple = samples.next()
exampple


def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

res = preprocess_twin(*exampple)

plt.imshow(res[1])

res[2]

# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

# Training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

inp = Input(shape=(100,100,3), name='input_image')
c1 = Conv2D(64, (10,10), activation='relu')(inp)
m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
c2 = Conv2D(128, (7,7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
c3 = Conv2D(128, (4,4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
c4 = Conv2D(256, (4,4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(4096, activation='sigmoid')(f1)
mod = Model(inputs=[inp], outputs=[d1], name='embedding')
mod.summary()
