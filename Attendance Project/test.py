from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import csv
import time
from datetime import datetime
import os

# Load trained data and models
traineddata = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

with open("data/names.pkl", "rb") as f:
    LABELS = pickle.load(f)

with open("data/faces_data.pkl", "rb") as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ["NAMES", "TIME"]

while True:
    success, frame = video.read()
    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = traineddata.detectMultiScale(gray_video)
    print(faces)

    for x, y, w, h in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        filename = "Attendance/Attendance_" + date + ".csv"
        exist = os.path.isfile(filename)

        attendance = [str(output[0]), str(timestamp)]

        # Check if the student has already marked attendance
        already_marked = False
        if exist:
            with open(filename, "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] == str(output[0]):
                        already_marked = True
                        break

        if already_marked:
            cv2.putText(frame, "Attendance already taken for "+output[0], (100, 570), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y - 65), (x + w, y), (50, 50, 255), -2)
            cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.putText(frame, "PRESS 'A' TO TAKE ATTENDANCE", (100, 570), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

            # Add attendance if 'A' is pressed and not already marked
            key = cv2.waitKey(1)
            if key == 97 and not already_marked:  # 'A' key pressed
                time.sleep(3)
                with open(filename, "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    if not exist:
                        writer.writerow(COL_NAMES)
                    writer.writerow(attendance)

    cv2.imshow("Video", frame)

    # Check if 'Q' is pressed to break the loop
    key = cv2.waitKey(1)
    if key == 113:  # 'Q' key pressed
        break

video.release()
cv2.destroyAllWindows()
