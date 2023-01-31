import mediapipe 
import cv2
import matplotlib.pyplot as plt
import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mediapipe
from keras_facenet import FaceNet
import pickle  
from sklearn.preprocessing import LabelEncoder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Facenet model extracting embddings from faces 
 
embedder = FaceNet()

def get_embedding(face_img):
    """
    Extract Embedding from faces 
    return Embedding on face to predict on them  
    """
    face_img = face_img.astype('float32') 
    face_img = np.expand_dims(face_img, axis=0) 
    yhat= embedder.embeddings(face_img)
    return yhat[0] 


 
# Load Are pretrained classifer 
pickle_modle = pickle.load(open("model_2.pkl", "rb"))

# Load Embeddings and labels of trained data 
nn = np.load("faces_embeddings_done_4classes.npz")
# pulling labels of trained data
y = nn['arr_1']


# encoding labels converting into numbers 
encoder = LabelEncoder()
encoder.fit(y)
# transform them into numbers 
y = encoder.transform(y)


# start Rolling camera Video capturing 
cam = cv2.VideoCapture(0)
 
# Mediapipe for face detecting faces 
face_detector_model = mediapipe.solutions.face_detection
# calling in detector faces with confidence > 0.75   
face_detector = face_detector_model.FaceDetection(min_detection_confidence=0.75)
 

while True :
    timer = cv2.getTickCount()
    # Reading in image from real time video 
    sucsses , img = cam.read()
    # converting reading image into RGB
    image_RGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    # Detecting face from coverted image 
    reusult2 = face_detector.process(image_RGB) 
    # if there is detected face then do the underline operations 
    if reusult2.detections:
        # for loop to extract the detected face bounding box -> height ,  width , channels of the captured image 
        for face_detected in reusult2.detections:
            ih , iw , ic = img.shape
            # extract the bouding box of detected face 
            bounding_box = face_detected.location_data.relative_bounding_box
            
            # extract boundary of detected face 
            x = int(bounding_box.xmin * iw)
            w = int(bounding_box.width * iw)
            y = int(bounding_box.ymin * ih)
            h = int(bounding_box.height * ih)
            
            # Resize or take area of intrest from whole image 
            CROPD_IMG = img[y:y+h , x:x+w]
            # take the resized image in which holds the area of intest and extract embedding 
            test_im = get_embedding(CROPD_IMG)
            # draw a rectangle around detect face 
            image = cv2.rectangle(img,(x,y),(x +w , y+h ),color = (255,0,0),thickness = 3 )
            # save extracted embeddings into a Variable to predict on it 
            test_im = [test_im]
            # predict on extracted embeddings with are classifer in which decside which person belongos to  
            ypreds = pickle_modle.predict(test_im)
            
    # show how much fps frame per second 
    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    # Put text on top of the rectangle identafiying how's the person is ! 
    cv2.putText(img =image, text=encoder.inverse_transform(ypreds)[0], org=(x, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),thickness=2)
   
    # SHOWING IMAGE FROM CAPTURED CAM 
    cv2.imshow('ORGINAL IMAGE' , img) 
    # SHOWING The  croped image
    cv2.imshow('croped IMAGE' , CROPD_IMG)  
 
    # TO ESC THE FRAME 
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
   
cam.release()
cv2.destoryAllWindows()

    