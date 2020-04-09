from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.engine.topology import Layer
from keras import backend as K
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utils import LRN2D
import utils
import pickle
import tkinter
from tkinter import *
from tkinter import messagebox
import math

#%load_ext autoreload
#%autoreload 2

np.set_printoptions(threshold=np.nan)

import speech_recognition as sr
import pyttsx3
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders

# Change these according to your credentials
fromaddr =  "mishtudeep.@gmail.com"
toaddr   =  "shreedeep.g@mefy.care" 
passwd   =  "Sreeshree123#"

##################################### VOICE RECOGNITION #########################################

def recognize():
    r = sr.Recognizer()
    r.pause_threshold = 0.7
    r.energy_threshold = 400
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    try:
        recognizedAudio =  r.recognize_google(audio)
        print("You said : " + recognizedAudio)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    return recognizedAudio

####################################### VOICE PROMPT ############################################
        
def speak(word):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)    
    engine.setProperty('volume',1.0)  
    voices = engine.getProperty('voices')   
    engine.setProperty('voice', voices[1].id)
    engine.say(word)
    engine.runAndWait()
    engine.stop()

######################################### SEND MAIL ##############################################

def sendMail():
    
    msg = MIMEMultipart()  
    msg['From'] = fromaddr 
    msg['To'] = toaddr 
    msg['Subject'] = "Mail test from Python end."
    body = "Hi, I am from MeFy. I am being tested by my creator." 
    msg.attach(MIMEText(body, 'plain')) 
    filename = "MeFy1.jpg"								  # Provide the file name
    attachment = open("C:\\Users\\SREEDEEP\\Documents\\face\\Mail sender\\MeFy1.jpg", "rb")    # Provide the complete address to the file within " "
    p = MIMEBase('application', 'octet-stream') 
    p.set_payload((attachment).read()) 
    encoders.encode_base64(p) 
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)  
    msg.attach(p) 
    s = smtplib.SMTP('smtp.gmail.com', port=587, timeout=25) 
    s.starttls() 
    s.login(fromaddr, passwd) 
    text = msg.as_string() 
    s.sendmail(fromaddr, toaddr, text) 
    s.quit()

###################################################################################################


myInput = Input(shape=(96, 96, 3))

x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
x = Activation('relu')(x)
x = ZeroPadding2D(padding=(1, 1))(x)
x = MaxPooling2D(pool_size=3, strides=2)(x)
x = Lambda(LRN2D, name='lrn_1')(x)
x = Conv2D(64, (1, 1), name='conv2')(x)
x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
x = Activation('relu')(x)
x = ZeroPadding2D(padding=(1, 1))(x)
x = Conv2D(192, (3, 3), name='conv3')(x)
x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
x = Activation('relu')(x)
x = Lambda(LRN2D, name='lrn_2')(x)
x = ZeroPadding2D(padding=(1, 1))(x)
x = MaxPooling2D(pool_size=3, strides=2)(x)

# Inception3a
inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
inception_3a_pool = Activation('relu')(inception_3a_pool)
inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

# Inception3b
inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

inception_3b_pool = Lambda(lambda x: x**2, name='power2_3b')(inception_3a)
inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
inception_3b_pool = Lambda(lambda x: x*9, name='mult9_3b')(inception_3b_pool)
inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
inception_3b_pool = Activation('relu')(inception_3b_pool)
inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

# Inception3c
inception_3c_3x3 = utils.conv2d_bn(inception_3b,
                                   layer='inception_3c_3x3',
                                   cv1_out=128,
                                   cv1_filter=(1, 1),
                                   cv2_out=256,
                                   cv2_filter=(3, 3),
                                   cv2_strides=(2, 2),
                                   padding=(1, 1))

inception_3c_5x5 = utils.conv2d_bn(inception_3b,
                                   layer='inception_3c_5x5',
                                   cv1_out=32,
                                   cv1_filter=(1, 1),
                                   cv2_out=64,
                                   cv2_filter=(5, 5),
                                   cv2_strides=(2, 2),
                                   padding=(2, 2))

inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

#inception 4a
inception_4a_3x3 = utils.conv2d_bn(inception_3c,
                                   layer='inception_4a_3x3',
                                   cv1_out=96,
                                   cv1_filter=(1, 1),
                                   cv2_out=192,
                                   cv2_filter=(3, 3),
                                   cv2_strides=(1, 1),
                                   padding=(1, 1))
inception_4a_5x5 = utils.conv2d_bn(inception_3c,
                                   layer='inception_4a_5x5',
                                   cv1_out=32,
                                   cv1_filter=(1, 1),
                                   cv2_out=64,
                                   cv2_filter=(5, 5),
                                   cv2_strides=(1, 1),
                                   padding=(2, 2))

inception_4a_pool = Lambda(lambda x: x**2, name='power2_4a')(inception_3c)
inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
inception_4a_pool = Lambda(lambda x: x*9, name='mult9_4a')(inception_4a_pool)
inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)
inception_4a_pool = utils.conv2d_bn(inception_4a_pool,
                                   layer='inception_4a_pool',
                                   cv1_out=128,
                                   cv1_filter=(1, 1),
                                   padding=(2, 2))
inception_4a_1x1 = utils.conv2d_bn(inception_3c,
                                   layer='inception_4a_1x1',
                                   cv1_out=256,
                                   cv1_filter=(1, 1))
inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

#inception4e
inception_4e_3x3 = utils.conv2d_bn(inception_4a,
                                   layer='inception_4e_3x3',
                                   cv1_out=160,
                                   cv1_filter=(1, 1),
                                   cv2_out=256,
                                   cv2_filter=(3, 3),
                                   cv2_strides=(2, 2),
                                   padding=(1, 1))
inception_4e_5x5 = utils.conv2d_bn(inception_4a,
                                   layer='inception_4e_5x5',
                                   cv1_out=64,
                                   cv1_filter=(1, 1),
                                   cv2_out=128,
                                   cv2_filter=(5, 5),
                                   cv2_strides=(2, 2),
                                   padding=(2, 2))
inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

#inception5a
inception_5a_3x3 = utils.conv2d_bn(inception_4e,
                                   layer='inception_5a_3x3',
                                   cv1_out=96,
                                   cv1_filter=(1, 1),
                                   cv2_out=384,
                                   cv2_filter=(3, 3),
                                   cv2_strides=(1, 1),
                                   padding=(1, 1))

inception_5a_pool = Lambda(lambda x: x**2, name='power2_5a')(inception_4e)
inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
inception_5a_pool = Lambda(lambda x: x*9, name='mult9_5a')(inception_5a_pool)
inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)
inception_5a_pool = utils.conv2d_bn(inception_5a_pool,
                                   layer='inception_5a_pool',
                                   cv1_out=96,
                                   cv1_filter=(1, 1),
                                   padding=(1, 1))
inception_5a_1x1 = utils.conv2d_bn(inception_4e,
                                   layer='inception_5a_1x1',
                                   cv1_out=256,
                                   cv1_filter=(1, 1))

inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

#inception_5b
inception_5b_3x3 = utils.conv2d_bn(inception_5a,
                                   layer='inception_5b_3x3',
                                   cv1_out=96,
                                   cv1_filter=(1, 1),
                                   cv2_out=384,
                                   cv2_filter=(3, 3),
                                   cv2_strides=(1, 1),
                                   padding=(1, 1))
inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
inception_5b_pool = utils.conv2d_bn(inception_5b_pool,
                                   layer='inception_5b_pool',
                                   cv1_out=96,
                                   cv1_filter=(1, 1))
inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

inception_5b_1x1 = utils.conv2d_bn(inception_5a,
                                   layer='inception_5b_1x1',
                                   cv1_out=256,
                                   cv1_filter=(1, 1))
inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
reshape_layer = Flatten()(av_pool)
dense_layer = Dense(128, name='dense_layer')(reshape_layer)
norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)


# Final Model
model = Model(inputs=[myInput], outputs=norm_layer)


# GRADED FUNCTION: triplet_loss

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(anchor - positive))
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(anchor - negative))
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.maximum(basic_loss, 0)
    ### END CODE HERE ###
    
    return loss
	
model.load_weights('model.h5')

def image_to_embedding(image, model):
    #image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA) 
    image = cv2.resize(image, (96, 96)) 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding
	
def recognize_face(face_image, input_embeddings, model):

    embedding = image_to_embedding(face_image, model)
    
    minimum_distance = 200
    name = None
    
    # Loop over  names and encodings.
    for (input_name, input_embedding) in input_embeddings.items():
        
       
        euclidean_distance = np.linalg.norm(embedding-input_embedding)
        #euclidean_distance = np.sum(np.square(embedding-input_embedding))

        print('Euclidean distance from %s is %s' %(input_name, euclidean_distance))

        
        if euclidean_distance < minimum_distance:
            minimum_distance = euclidean_distance
            name = input_name
    
    if minimum_distance < 0.70:
        return str(name[:-2])
    else:
        return str('Unknown')
		
try:
    pickle_in = open("dict.pickle","rb")
    database = pickle.load(pickle_in)
except:
    database={}
	
def most_common(lst):
    return max(set(lst),key=lst.count)
	
import glob

def create_input_image_embeddings():
    input_embeddings = {}

    for file in glob.glob("images/*"):
        person_name = os.path.splitext(os.path.basename(file))[0]
        image_file = cv2.imread(file, 1)
        input_embeddings[person_name] = image_to_embedding(image_file, model)

    return input_embeddings

def recognize_faces_in_cam(input_embeddings=database):
    

    cv2.namedWindow("Face Recognizer")
    vc = cv2.VideoCapture(0)
   

    font = cv2.FONT_HERSHEY_SIMPLEX
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    count=0
    lst=[]
    while vc.isOpened():
        _, frame = vc.read()
        img = frame
        height, width, channels = frame.shape

        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through all the faces detected 
        identities = []
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h

           
            
            face_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]    
            identity = recognize_face(face_image, input_embeddings, model)
            
            

            if identity is not None:
                img = cv2.rectangle(frame,(x1, y1),(x2, y2),(0,255,0),2)
                cv2.putText(img, str(identity), (x1+5,y1-5), font, 1, (0,255,0), 2)
                lst.append(identity)
        count+=1
            
        key = cv2.waitKey(100)
        cv2.imshow("Face Recognizer", img)
        if(count>=10):
            break
        elif key == 27: # exit on ESC
            break
    vc.release()
    cv2.destroyAllWindows()
    if(len(lst)!=0):
        user=most_common(lst)
    else:
        user='noface'
    return user
	
def captureImages():
    
    e_id=eid_entry.get()
    e_name=name_entry.get()
    email=email_entry.get()
    speak("Hello "+e_name+". Welcome to face recognition system. We are going to take some photos. Get ready.")
    cam = cv2.VideoCapture(0)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    count = 0
    while(True):
        ret, img = cam.read()
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), 2)     
        # Save the captured image into the datasets folder
            path="images/"+e_id+"_" + str(count) + ".jpg"
            cv2.imwrite(path, img[y1:y2,x1:x2])
            cv2.imshow('image', img)
        #input_embeddings[user]=image_to_embedding(cv2.imread(path),model)
            count += 1
        k = cv2.waitKey(200) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 10: # Take 30 face sample and stop video
             break
    cam.release()
    cv2.destroyAllWindows()
    for file in glob.glob("images/"+e_id+"_*"):
            person_name = os.path.splitext(os.path.basename(file))[0]
            image_file = cv2.imread(file, 1)
            database[person_name] = image_to_embedding(image_file, model)
    pickle_out = open("dict.pickle","wb")
    pickle.dump(database, pickle_out)
    pickle_out.close()
    for sheet_no in range(12):
        sheet=client.open("Attendance System").get_worksheet(sheet_no)
        sheet.insert_row([e_id,e_name,email],5)
    speak("Successfully registered")
	

import datetime
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
credentials=ServiceAccountCredentials.from_json_keyfile_name('Attendance-72970c144e04.json',scope)
client=gspread.authorize(credentials)

window=Tk()
window.title("Face Recognition")
window.resizable(0,0)
banner=tkinter.PhotoImage(file='C:/Users/Sourav/PROJECTS/Face Recognition/Face-recognition (implementation of Facenet)/facial-recognition-ibm-1440x920.gif')
#tkinter.Label(window,image=banner) .grid(row=0,column=0,sticky=E)
w = banner.width()
h = banner.height()

# size the window so the image will fill it
window.geometry("%dx%d+50+30" % (w, h))

cv = tkinter.Canvas(width=w, height=h)
cv.pack(side='top', fill='both', expand='yes')
cv.create_image(0, 0, image=banner,anchor='nw')
# add canvas text at coordinates x=15, y=20
# anchor='nw' implies upper left corner coordinates
cv.create_text(20, 20, text="Face Recognition Attendance System",font=("agency fb bold",28), fill="white", anchor='nw')
cv.create_text(20, 90, text="New Registration",font=("agency fb bold",22), fill="white", anchor='nw')

cv.create_text(20, 130, text="Enter your name:",font=("agency fb bold",14), fill="white", anchor='nw')
name_entry=Entry(cv, width=40,bg='white')
name_entry.place(x=20,y=160)

cv.create_text(20, 210, text="Enter your employee ID:",font=("agency fb bold",14), fill="white", anchor='sw')
eid_entry=Entry(cv, width=40,bg='white')
eid_entry.place(x=20,y=220)

cv.create_text(20, 270, text="Enter your email ID:",font=("agency fb bold",14), fill="white", anchor='sw')
email_entry=Entry(cv, width=40,bg='white')
email_entry.place(x=20,y=280)


def register():
    if (name_entry.get() and eid_entry.get() and email_entry.get()):
        captureImages()
    else:
        speak("Please enter your details in the text box")
        #msg=messagebox.showinfo("Error","Please enter your name in the text box")
        
def checkIn():
    date_time=datetime.datetime.now().isoformat()
    if((int(date_time[11:13])+int(date_time[14:16])/60)>16):
            speak("Sorry. Time period to enter into the office has exceeded.")
    else:
        user=recognize_faces_in_cam(database)
        sheet_no=int(date_time[5:7])-1
        sheet=client.open("Attendance System").get_worksheet(sheet_no)
        try:
            row_no=sheet.col_values(1).index(user) + 1
            col_no=(int(date_time[8:10]))*6-2
            entry=sheet.cell(row_no,col_no).value.split(':')
        except:
            print("no record found in sheets")
        
        try:
            if(user=='Unknown'):
                speak("Sorry. No record found. If you are a new user, please register yourself.")
            elif(user=='noface'):
                speak("Sorry. No face detected. Try again.")
            elif(entry[0]!=''):
                speak("Hi "+sheet.cell(row_no,2).value+", you have already checked in at"+entry[0]+"hours"+entry[1]+"minutes.")
            else:
                sheet.update_cell(row_no,col_no,date_time[11:])
                speak("Hi "+sheet.cell(row_no,2).value+", Welcome.")
        except:
            speak("Something went wrong. Please Try again.")
        
def checkOut():
    user=recognize_faces_in_cam(database)
    try:
        date_time=datetime.datetime.now().isoformat()
        sheet_no=int(date_time[5:7])-1
        sheet=client.open("Attendance System").get_worksheet(sheet_no)
        try:
            row_no=sheet.col_values(1).index(user) + 1
            col_no=(int(date_time[8:10]))*6-1
            entry=sheet.cell(row_no,col_no-1).value.split(':')
            min_exit_time=float((int(entry[0])+int(entry[1])/60)+0)
            exit=sheet.cell(row_no,col_no).value.split(':')
        except:
            print("no record found in sheets")
        if(user=='Unknown'):
            speak("Sorry. No record found. If you are a new user, please register yourself.")
        elif(user=='noface'):
            speak("Sorry. No face detected. Try again.")
        elif(entry[0]==''):
            speak("Hi "+sheet.cell(row_no,2).value+", you did not check in yet today.")
        elif((int(date_time[11:13])+int(date_time[14:16])/60)<min_exit_time):
            speak("Sorry. You can not leave before "+str(int(min_exit_time))+"hours"+str(math.ceil((min_exit_time*60)%60))+"minutes")
        elif(exit[0]!=''):
            speak("Hi "+sheet.cell(row_no,2).value+", you have already checked out at"+exit[0]+"hours"+exit[1]+"minutes.")
        else:
            sheet.update_cell(row_no,col_no,datetime.datetime.now().isoformat()[11:])
            speak("Good bye Mr. "+sheet.cell(row_no,2).value)
    except:
        speak("Something went wrong. Please Try again.")
        
def midOut():
    user=recognize_faces_in_cam(database)
    try:
        date_time=datetime.datetime.now().isoformat()
        sheet_no=int(date_time[5:7])-1
        sheet=client.open("Attendance System").get_worksheet(sheet_no)
        try:
            row_no=sheet.col_values(1).index(user) + 1
            col_no=(int(date_time[8:10]))*6
            entry=sheet.cell(row_no,col_no-2).value.split(':')
            exit=sheet.cell(row_no,col_no-1).value.split(':')
        except:
            print("no record found in sheets")
        if(user=='Unknown'):
            speak("Sorry. No record found. If you are a new user, please register yourself.")
        elif(user=='noface'):
            speak("Sorry. No face detected. Try again.")
        elif(entry[0]==''):
            speak("Hi "+sheet.cell(row_no,2).value+", you did not check in yet today.")
        elif(exit[0]!=''):
            speak("Hi "+sheet.cell(row_no,2).value+", you have already checked out for today at"+exit[0]+"hours"+exit[1]+"minutes.")
        else:
            sheet.update_cell(row_no,col_no,datetime.datetime.now().isoformat()[11:])
            speak("Hi Mr. "+sheet.cell(row_no,2).value+", please return to the office within thirty minutes.")
    except:
        speak("Something went wrong. Please Try again.")
        

def midIn():
    date_time=datetime.datetime.now().isoformat()
    user=recognize_faces_in_cam(database)
    sheet_no=int(date_time[5:7])-1
    sheet=client.open("Attendance System").get_worksheet(sheet_no)
    try:
        row_no=sheet.col_values(1).index(user) + 1
        col_no=(int(date_time[8:10]))*6+1
        entry=sheet.cell(row_no,col_no-3).value
        mid_out=sheet.cell(row_no,col_no-1).value
        exit=sheet.cell(row_no,col_no-2).value.split(':')
    except:
        print("no record found in sheets")
        
    try:
        if(user=='Unknown'):
            speak("Sorry. No record found. If you are a new user, please register yourself.")
        elif(user=='noface'):
            speak("Sorry. No face detected. Try again.")
        elif(entry==''):
            speak("Hi "+sheet.cell(row_no,2).value+", you did not check in yet today.")
        elif(exit[0]!=''):
            speak("Hi "+sheet.cell(row_no,2).value+", you have already checked out for today at"+exit[0]+"hours"+exit[1]+"minutes.")
        elif(mid_out==''):
            speak("Hi "+sheet.cell(row_no,2).value+", you did not mid check out today")
        else:
            sheet.update_cell(row_no,col_no,date_time[11:])
            speak("Hi "+sheet.cell(row_no,2).value+", Welcome.")
    except:
        print(error)
        speak("Something went wrong. Please Try again.")
            
def voice():
    speak("How may I help you?")
    word=recognize()
    if(word=="I want to check in"):
        checkIn()
    elif(word=="I want to check out"):
        checkOut()
    elif(word=="I want to mid in"):
        midIn()
    elif(word=="I want to mid out"):
        midOut()
    else:
	    speak("Please give valid command.")
    
btn1 = tkinter.Button(cv, text="Register",font=('roboto',10),bg='#48bec5',fg='white',command=register)
btn1.place(x=20,y=315)

btn3 = tkinter.Button(cv, text="CHECK OUT",font=('roboto',12),bg='#48bec5',fg='white',command=checkOut)
btn3.pack(anchor=SE,side='right',ipadx=15,ipady=10,padx=10,pady=60)

btn4 = tkinter.Button(cv, text="MID OUT",font=('roboto',12),bg='#48bec5',fg='white',command=midOut)
btn4.pack(anchor=SE,side='right',ipadx=15,ipady=10,padx=10,pady=60)

btn5 = tkinter.Button(cv, text="MID IN",font=('roboto',12),bg='#48bec5',fg='white',command=midIn)
btn5.pack(anchor=SE,side='right',ipadx=15,ipady=10,padx=10,pady=60)

btn2 = tkinter.Button(cv, text="CHECK IN",font=('roboto',12),bg='#48bec5',fg='white',command=checkIn)
btn2.pack(anchor=SE,side='right',ipadx=15,ipady=10,padx=10,pady=60)

photo = PhotoImage(file='microphone-512.png').subsample(15,15)
btn6 = Button(cv, image=photo, bd=0, bg='#48bec5', overrelief='groove', relief='sunken',command=voice)
btn6.place(x=325,y=280,height=60, width=60)


window.mainloop()
