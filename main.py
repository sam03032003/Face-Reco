import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'ImageAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList :
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)


# func for encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []

        print(myDataList)
        for line in myDataList:
            entry = line.split(' , ')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name} , {dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None , 0.25 , 0.25)
    imgS = cv2.cvtColor(imgS , cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS , faceCurFrame)

    for encodeFace , faceLoc in zip(encodeCurFrame , faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown , encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown , encodeFace)
        # print(faceDis)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4 , x2*4 , y2*4 , x1*4

            cv2.rectangle(img , (x1,y1) , (x2,y2) , (0,255,0) , 2)
            cv2.rectangle(img, (x1,y2-35) , (x2,y2) , (0,255,0) , cv2.FILLED)
            cv2.putText(img , name , (x1+6 , y2-6) , cv2.FONT_HERSHEY_TRIPLEX , 1 , (255,255,255) , 2)
            markAttendance(name)


    cv2.imshow('Webcam' , img)
    cv2.waitKey(1)






# # Step 1 :- Loading the image and converting it into RGB image
#
# img1 = face_recognition.load_image_file('ImageBasic/Sam1.jpg')
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)       # Converting our BGR image to RGB image
#
#             # Now Test Image
# imgTest = face_recognition.load_image_file('ImageBasic/Sam2.jpg')
# imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)       # Converting our BGR image to RGB image
#
#
#
# # Step 2 :- Finding the faces in the image and finding their encodings in that
#
# faceLoc = face_recognition.face_locations(img1)[0]      # Detect's face
# encodeImg1 = face_recognition.face_encodings(img1)[0]   # encoding face
# cv2.rectangle(img1 , (faceLoc[3] , faceLoc[0]) , (faceLoc[1] , faceLoc[2]) , (255,0,255) , 2 )
#
# faceLocTest = face_recognition.face_locations(imgTest)[0]      # Detect's face
# encodeImgTest = face_recognition.face_encodings(imgTest)[0]   # encoding face
# cv2.rectangle(imgTest , (faceLocTest[3] , faceLocTest[0]) , (faceLocTest[1] , faceLocTest[2]) , (255,0,255) , 2 )
#
#
#
# # Step 4 :- Comparing both faces by encodings
#
# resuls = face_recognition.compare_faces([encodeImg1], encodeImgTest)
# faceDis = face_recognition.face_distance([encodeImg1], encodeImgTest)
# print(resuls , faceDis)
#
# cv2.imshow('Sahil Original', img1)
# cv2.imshow('SahilTest', imgTest)
# cv2.waitKey(0)
