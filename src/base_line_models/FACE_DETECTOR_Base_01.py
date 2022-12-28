import mediapipe
import cv2
import time
# in reallty i need to now each point how to handle them which i have over 480 landmarks
# so that a large number so i need to look at them 
# idenfiy which is revier to in case of features 
# starting of eye or leeps and edge | FACE DETECTOR & FACE MESH

# FACE Detector Mesh ->>>>> Base Line
class FaceDetectorMesh():
    def __init__(self , static_image = False , max_faces = 1 , refine_landmarks = False ,
                     min_detection_confidence = 0.5, min_tracking_confidence=0.5 ):
        # intial for refering to init parametres 
        self.static_image = static_image
        self.max_faces = max_faces 
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # calling drawing from mediapipe 
        self.mediapipe_drawing = mediapipe.solutions.drawing_utils
        # CALLING FACE MESH FROM MEDIAPIPE
        self.mediapipe_facemesh = mediapipe.solutions.face_mesh
        # OBJECT FORM FACE MESH | including parametres that user could change inside face mesh hyper parmaters
        self.face_mesh = self.mediapipe_facemesh.FaceMesh(self.static_image,self.max_faces,self.refine_landmarks, 
                                                          self.min_detection_confidence,
                                                          self.min_tracking_confidence)
        # ADJUSTING THICKNESS AND RADIUS OF LAND MARKES 
        self.draw_specs = self.mediapipe_drawing.DrawingSpec(thickness = 1 ,circle_radius = 2 )
        
        
    def findFaceMesh(self , img , draw = True):
        self.image_RGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        # REUSLT OF FACE MESH ON RGB IMAGE
        self.RESULT = self.face_mesh.process(self.image_RGB)
        # drawing the result on faces 
        # if result is valid with face landmarks 
        # all faces which represent all value of land mark evey time the face it's detected 
        faces = []
        # if result processing the image is true or falled then continue 
        if self.RESULT.multi_face_landmarks:
            # take each landmarks on faces from multi face landmarks
            for facelanmark  in self.RESULT.multi_face_landmarks:
                # drawing landmarks on the face detectd 
                if draw:
                    self.mediapipe_drawing.draw_landmarks(img ,facelanmark ,
                                                 self.mediapipe_facemesh.FACEMESH_CONTOURS ,
                                                 self.draw_specs , self.draw_specs )
                
              #another for loop to get number of all land markas to get deep 
              #knowing start and edges of land marks 
                # list of face detected 
                face = []
                # for loop which takes lm -> land marks value out of 480 land marks , 
                # id which represent -> the id of the land mark for each lm -> id which represent if it's -> right eye 
                # etc...
                for id,lm in enumerate(facelanmark.landmark):
                    # handle normalized number by multiple by x * width and y * height
                    ih , iw , ic = img.shape
                    x , y = (int(lm.x*iw) , int(lm.y*ih))
                    # appending x , y of the face detected
                    face.append([x,y])
                faces.append(face)
                    
                    #furure more need to save all in a list 
                    #which will be done in a module will be much faster
            
        return img , id , faces
        
        




def main():
    # start the camera to caputre video 
    cam = cv2.VideoCapture(0)
    # previous time stating value 
    Ptime = 0 
    # creating object from class face mesh detector
    detector = FaceDetectorMesh()
    while True:
        # from camera capture frames and image
        red , img = cam.read()
        # start detecting the face using face mesh calling them from detector moudle
        img ,id ,faces = detector.findFaceMesh(img)
        # showing the faces land marks values also there id 
        print(id , faces)
        # current time running. 
        Ctime = time.time()
        # fps 
        # 1 / current_time - previos_time 
        fps = 1/(Ctime - Ptime)
        # current equal to previos_time
        Ptime = Ctime
        # putext (image , text , location , fontstyle , scale , color , thicknees)
        cv2.putText(img , 'FPS : {}'.format(int(fps)) , (20,70) , cv2.FONT_HERSHEY_COMPLEX
                   , 3 , (0,255,255) , thickness = 3)
        # SHOWING IMAGE FROM CAPTURED CAM 
        cv2.imshow('ORGINAL IMAGE' , img)
        # ESC | EXIT FROM THE FRAME SHOUTDOWN CAM | Frame
        if cv2.waitKey(100) == 27:
            break 
    cam.release()
    cv2.destoryAllWindows()





if __name__ == '__main__':
    main()