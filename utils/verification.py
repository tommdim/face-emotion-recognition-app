import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import uuid
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from nuovavita import L1Dist

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

class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

siamese_model = tf.keras.models.load_model('siamesemodelv2.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

models = ['siamesemodelv2.h5','tommasosiamesemodelv2.h5','federicasiamesemodelv2.h5','elenasiamesemodelv2.h5']
dirs = ['verification_images','t_verification_images','f_verification_images','e_verification_images']
dic = {}

def verify(model, detection_threshold, verification_threshold, ver_img):
    # Build results array
    results = []
    
    for image in os.listdir(os.path.join('application_data', ver_img)):
        if image==".DS_Store":
            os.remove(os. path. join('application_data', ver_img+'/.DS_Store'))
            print('merda removed')
        else:
            input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = preprocess(os.path.join('application_data', ver_img, image))
            
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    

    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('application_data', ver_img))) 
    
    # verified = verification > verification_threshold
    
    return verification
def capture():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[200:200+250,570:570+250, :]    
        cv2.imshow('Verification', frame)
    
        # Verification trigger
        if cv2.waitKey(10) & 0xFF == ord('v'):


            cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
            # Run verificationv
            for index in range(len(models)):
            
                siamese_model = tf.keras.models.load_model(models[index], 
                                    custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy}, compile = False)
                verification_img = dirs[index]
                # verification_img = 'verification_images'
                # input_img = 'input_images'
                verification = verify(siamese_model, 0.7, 0.7,verification_img)
                dic[models[index]] = verification
        
            for key in dic.keys():
                if dic[key] == max(dic.values()):
                    if dic[key]>0.7:
                
                        if key == 'siamesemodelv2.h5':
                            print('sei Vale!!!')
                        if key == 'tommasosiamesemodelv2.h5':
                            print('sei Tommi!!!')
                        if key == 'elenasiamesemodelv2.h5':
                            print ('sei Ele!!!')
                        if key == 'federicasiamesemodelv2.h5':
                            print('sei Fede!!!')
                    else:
                        print('Bro non so chi sei')
            print(dic)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

capture()

