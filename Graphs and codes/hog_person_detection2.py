import cv2
import time
import imutils
import matplotlib.pyplot as plt 

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture('therm.mp4')
#cap.set(cv2.cv.CV_CAP_PROP_FPS,60)
#cap = cv2.imread('img_00001.bmp')
img_array=[]
list_times=[]
frame_no=[]
i=0
while True:
    r, frame = cap.read()
    if r:

        start_time = time.time()
        frame = imutils.resize(frame,1000,500) 
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # HOG needs a grayscale image

        rects, weights = hog.detectMultiScale(gray_frame)
        
        # Measure elapsed time for detections
        end_time = time.time()
        diff=end_time-start_time
        print("Elapsed time:", diff)
        list_times.append(diff)
        frame_no.append(i)
        i=i+1
        
        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.7:
                continue
            cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)

        height, width, layers = frame.shape
        size = (width,height)
        img_array.append(frame)
    else:
        break
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
         out.write(img_array[i])
         cv2.imshow("preview", frame)
    out.release()
    k = cv2.waitKey(1)
    #if k & 0xFF == ord("q"): # Exit condition
     #   break


# plotting the points  
plt.plot(frame_no, list_times) 
  
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
  
# giving a title to my graph 
plt.title('TIME ELAPSED-HOG') 
  
# function to show the plot 
plt.show() 
