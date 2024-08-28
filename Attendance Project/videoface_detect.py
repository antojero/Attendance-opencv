import cv2
import pickle
import numpy as np
import os

traineddata = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

faces_data = []
i=0

name = input("Enter Your name")

while True:
    success, frame = video.read()
    if success == True:
        gray_video = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = traineddata.detectMultiScale(gray_video)
        print(faces)

        for x, y, w, h in faces:
            crop_img = frame[y:y+h,x:x+w,:]
            resized_img = cv2.resize(crop_img,(50,50))
            if len(faces_data)<=100 and i%10 == 0:
                faces_data.append(resized_img)
            i=i+1
            cv2.putText(frame,str(len(faces_data)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.imshow("Video",frame)
        key = cv2.waitKey(1)
        if key ==87 or key==113 or len(faces_data)==100:
            break
video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100,-1)

if 'names.pkl' not in os.listdir("data/"):
    names = [name]*100
    with open("data/names.pkl", "wb") as f:
        pickle.dump(names, f)

else:
    with open("data/names.pkl","rb") as f:
        names=pickle.load(f)
    names = names+[name]*100
    with open("data/names.pkl","wb") as f:
        pickle.dump(names,f)

if 'faces_data.pkl' not in os.listdir("data/"):
    with open("data/faces_data.pkl","wb") as f:
        pickle.dump(faces_data, f)

else:
    with open("data/faces_data.pkl","rb") as f:
        faces=pickle.load(f)
    faces = np.append(faces,faces_data,axis=0)
    with open("data/faces_data.pkl","wb") as f:
        pickle.dump(faces,f)
