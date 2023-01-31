import cv2 as cv
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import mediapipe
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Face Loadings Class 

class FACELOADING:

    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector_model = mediapipe.solutions.face_detection
        self.detector = self.detector_model.FaceDetection(min_detection_confidence=0.6)
 
    # this code to perform face detection fro all images before 
    # pushing them into the model in the way of retraingig or Traformating learning -> based on giving images 
    # for face deteaction consadration i'll be using face detector by ->> Mediapipe 
    def extract_face(self, filename):
        # Reading images 
        img = cv.imread(filename)
        # transformaing images into RGB in the way media pipe model accept RGB Color 
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
  
        # in this case take the width & height from the image 
        result = self.detector.process(img)
        # if there detection in which model detect the face for any image then this will give me the
        if result.detections:

          for face in result.detections:
            # then for each face detected with in 85% so process the under lines of code  
             if face.score[0] > 0.85 : 
              # boundary_box ->> containg face coardinate x ,y , w , h  
               bounding_box = face.location_data.relative_bounding_box

              #face coardinate x ,y , w , h -> represnting  in normalized manner 
              # in this case take the width & height from the image 
               width = img.shape[1]
               height = img.shape[0]
               # x min  * width -> turn to integer 
               x = int(bounding_box.xmin * width)
               # width with in bounding box * width -> turn to integer 
               w = int(bounding_box.width * width)
               # y min * height -> turn to integer 
               y = int(bounding_box.ymin * height)
               # height with in bounding box * ->|height turn to integer 
               h = int(bounding_box.height * height)  
               # just give me the face from over all image 
               face = img[y:y+h, x:x+w]
               # Resize the image into size of (160 , 160)
               face_arr = cv.resize(face, self.target_size)
               # return all resized face 
               return face_arr
    

    # loading faces from files or diractory | sub-diractory ->> 
    def load_faces(self, dir):
        FACES = []
        # extract image name form diractory where images are stored in .. 
        for im_name in os.listdir(dir):
          # try under line code 
            try:
                # give me images path contining diractory + image_name
                path = dir + im_name
                # then extract face from images using extract face method then store them into varibel 
                single_face = self.extract_face(path)
                # take extracted face into list named faces in which holds all extracted faces as resized array of each image 
                FACES.append(single_face)
                # if failled then pass return the faces list to the mehtod 
            except Exception as e:
                pass
        return FACES
    
    # extract class labels for the images which hold the first text of image name 
    def load_classes(self):
      # for sub diractores which are in main directory take each one  
        for sub_dir in os.listdir(self.directory):
            # path of sub diractory in which could extract images if they are in sub_dir 
            path = self.directory +'/'+ sub_dir+'/'
            # push the path into load faces method 
            FACES = self.load_faces(path)
            # extract image labels 
            labels = [sub_dir for _ in range(len(FACES))]
            # print if all images on the sub_dir are loaded 
            print(f"Loaded successfully: {len(labels)}")
            # push detected and resized faces into X varible 
            self.X.extend(FACES)
            # push extracted calsses into Y varible 
            self.Y.extend(labels)
        # return them as numpy array for faster and more efficent to store data 
        return np.asarray(self.X), np.asarray(self.Y)

    
    def plot_images(self):
        plt.figure(figsize=(18,16))
         
        for num,image in enumerate(self.X):
            if image is not None:
              ncols = 3
              nrows = len(self.Y)//ncols + 1
              plt.subplot(nrows,ncols,num+1)
              plt.imshow(image)
              plt.axis('off')





# Loading Faces .. 
faceloading = FACELOADING("/content/drive/MyDrive/traning_images") #  directory path  
X, Y = faceloading.load_classes() # loadings X representing detected faces , y represent label of each image in are case name of the person 




# Visulizing detected faces 
faceloading.plot_images()




# Extract embeding from detected faces which are way of recogniton by using embedings differnces .. 
embedder = FaceNet()

def get_embedding(face_img):
    face_img = face_img.astype('float64') # 3D(160x160x3) 
    face_img = np.expand_dims(face_img, axis=0)  
    # 4D (Nonex160x160x3)
    yhat= embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)



# For each detect face in  x  extract embedding of each  face and append to a list of embeddings 

EMBEDDED_X = []

for img in X:
  if img is not None:
     EMBEDDED_X.append(get_embedding(img))

EMBEDDED_X = np.asarray(EMBEDDED_X)




# Comperesing Emeddeings and labels to save them .. 
np.savez_compressed('faces_embeddings_done_4classes.npz', EMBEDDED_X, Y)




# Visulazing value counts of each label in are case it's name of each person 
pd.DataFrame(Y).value_counts().sort_values().plot(kind = 'barh' , figsize = (20,10))



# Encoding names of person into number before pusing them into a classifier 
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)


# Seting Y labels as same as the extracted embedding shape to split are them after 
Y = Y[:len(EMBEDDED_X)]


# Spliting the data into traning and testing sets before training are classifier 
X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y ,test_size= 0.20 , random_state=0)
 




 # Create are classifier which is Support vector classifer 
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)


# Predicting on train sampels and test sample 
ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)




# Measuring accuracy 
accuracy_score(y_pred  = ypreds_test, y_true = Y_test) * 100 


# Make prediction on new picture to test the model will recognize if the persons is the same or not 
resized_img  = faceloading.extract_face("/content/photo.jpg") 
# Show the image 
plt.imshow(resized_img)
# Extract the embedding of the new image 
test_im = get_embedding(resized_img)

# Push embedding into a list 
test_im = [test_im]
# Predict on this embedding of new image how is the person ? .. 
ypreds =  model.predict(test_im)


encoder.inverse_transform(ypreds)[0] # show the real name before it's encodined 


# saving the model 
pickle.dump(model, open('model_01.pkl', 'wb'))