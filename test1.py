import cv2
from tracker import*
import cvzone
import numpy as np
import os
from datetime import datetime
# Initialize the video capture
cap = cv2.VideoCapture('wrongwayfinal.mp4')  # Replace 'your_video.mp4' with your video file path

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()
area1=[(95,314),(456,369),(473,332),(128,299)]
area2=[(147,290),(476,323),(488,303),(165,282)]
tracker=Tracker()
a1={}
counter=[]

def save_full_frame(frame):
    # Create a folder with the current date and time as the folder name
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_name = f"wrongway"
    os.makedirs(folder_name, exist_ok=True)

    # Save the entire frame
    image_filename = os.path.join(folder_name, f"frame_{current_datetime}.jpg")
    cv2.imwrite(image_filename, frame)
while True:
    ret, frame = cap.read()
        
    if not ret:
        break
    frame=cv2.resize(frame,(1020,500))
    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Threshold the foreground mask
    _, thresh = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around moving objects
    list=[]
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            list.append([x,y,w,h])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x1,y1,w1,h1,id=bbox
        cx=int(x1+x1+w1)//2
        cy=int(y1+y1+h1)//2
        result=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
        if result>=0:
           a1[id]=(cx,cy)
        if id in a1:
           result1=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
           if result1>=0:
              cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
              cv2.circle(frame,(cx,cy),6,(0,255,0),-1)
              cvzone.putTextRect(frame,f'{id}',(x1,y1),1,1)
              if counter.count(id)==0:
                 counter.append(id)
                 save_full_frame(frame)


    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)
    p=len(counter)
    cvzone.putTextRect(frame,f'WrongsideVehicle:-{p}',(50,60),2,2)
    cv2.imshow('Motion Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
