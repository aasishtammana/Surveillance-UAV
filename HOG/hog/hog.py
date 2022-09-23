import cv2
import time
import imutils 

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture('therm.mp4')
#cap.set(cv2.cv.CV_CAP_PROP_FPS,60)
#cap = cv2.imread('img_00001.bmp')
img_array=[]
while True:
    r, frame = cap.read()
    if r:
        start_time = time.time()
        frame = imutils.resize(frame,width=1000,height=500) # Downscale to improve frame rate
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # HOG needs a grayscale image

        rects, weights = hog.detectMultiScale(gray_frame)
        
        # Measure elapsed time for detections
        end_time = time.time()
        print("Elapsed time:", end_time-start_time)
        
        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.7:
                continue
            cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)

        #cv2.imshow("preview", frame)
        height, width, layers = frame.shape
        size = (width,height)
        img_array.append(frame)
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
         cv2.imshow("image",frame)
         out.write(img_array[i])
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"): # Exit condition
        break

##
##import cv2 
##import imutils 
##   
### Initializing the HOG person 
### detector 
##hog = cv2.HOGDescriptor() 
##hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
##   
##cap = cv2.VideoCapture('soccer.avi') 
##while cap.isOpened(): 
##    # Reading the video stream 
##    ret, image = cap.read() 
##    if ret: 
##        image = imutils.resize(image,  
##                               width=min(400, image.shape[1])) 
##   
##        # Detecting all the regions  
##        # in the Image that has a  
##        # pedestrians inside it 
##        (regions, _) = hog.detectMultiScale(image, 
##                                            winStride=(4, 4), 
##                                            padding=(4, 4), 
##                                            scale=1.05) 
##   
##        # Drawing the regions in the  
##        # Image 
##        for (x, y, w, h) in regions: 
##            cv2.rectangle(image, (x, y), 
##                          (x + w, y + h),  
##                          (0, 0, 255), 2) 
##   
##        # Showing the output Image 
##        cv2.imshow("Image", image) 
##        if cv2.waitKey(25) & 0xFF == ord('q'): 
##            break
##    else: 
##        break
##  
##cap.release() 
##cv2.destroyAllWindows() 
##
##
##import numpy as np
##import cv2
## 
### initialize the HOG descriptor/person detector
##hog = cv2.HOGDescriptor()
##hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
##
##cv2.startWindowThread()
##
### open webcam video stream
##cap = cv2.VideoCapture('soccer.avi')
##
### the output will be written to output.avi
##out = cv2.VideoWriter(
##    'output.avi',
##    cv2.VideoWriter_fourcc(*'MJPG'),
##    15.,
##    (640,480))
##
##while(True):
##    # Capture frame-by-frame
##    ret, frame = cap.read()
##
##    # resizing for faster detection
##    frame = cv2.resize(frame, (500, 500))
##    # using a greyscale picture, also for faster detection
##    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
##
##    # detect people in the image
##    # returns the bounding boxes for the detected objects
##    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
##
##    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
##
##    for (xA, yA, xB, yB) in boxes:
##        # display the detected boxes in the colour picture
##        cv2.rectangle(frame, (xA, yA), (xB, yB),
##                          (0, 255, 0), 2)
##    
##    # Write the output video 
##    out.write(frame.astype('uint8'))
##    # Display the resulting frame
##    cv2.imshow('frame',frame)
##    if cv2.waitKey(1) & 0xFF == ord('q'):
##        break
##
### When everything done, release the capture
##cap.release()
### and release the output
##out.release()
### finally, close the window
##cv2.destroyAllWindows()
##cv2.waitKey(1)
