import cv2
import numpy as np 

#video capture
camera=cv2.VideoCapture("OaklandFireTherm.mp4")
res,frame_1=camera.read() #this is the first frame, res is a boolean to check if the frame is retrieved or not

#global variables
frame_width=camera.get(3) #get the width ,where 3  is a camera property for width
frame_height=camera.get(4) #get the height where 4 is a camera property for height
frame_size=(frame_height*frame_width) #the total frame size

print("Frame width =", frame_width)
print("Frame height =",frame_height)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() #used in background subtraction

#functions
def display(window_name,frame_name): #function displays output in a separate window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) #name the window
    cv2.resizeWindow(window_name,640, 480) #resize the window to the required resolution
    cv2.imshow(window_name,frame_name) #output of the image/frame/processed video

while(True):
    res,img=camera.read() #for the second frame
    
    if not res:
        text="No Video is captured"
        break

    #exception handling 
    try:
        pass
        #Input video file is given as output
    except cv2.error:
        print("There is an error printing the input image")

    blur = cv2.GaussianBlur(img, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [5, 50, 50]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)

    fireoutput = cv2.bitwise_and(img, hsv, mask=mask)
    no_red = cv2.countNonZero(mask)
    cv2.imshow("fire output", fireoutput)

    # rgb to grayscale conversion
    gray=cv2.cvtColor(fireoutput,cv2.COLOR_BGR2GRAY) #reducing the complexity of the algorithm since just dealing with just 1 channel
    # cv2.imshow("grayscale image",gray)

    # binary thresholding
    ret, th3 = cv2.threshold(gray ,100,255,cv2.THRESH_BINARY)
    cv2.imshow("binary threshold image",th3)

    # morphological image processing -----------------------------------------------------------------------------
    kernel_3 = np.ones((5,5),np.uint8)#5x5 kernel
    dilate=cv2.dilate(th3,kernel_3,iterations=2)#using the 3X3 kernel perform dilation twice so that we can clearly identify the object of interest in the image
    # cv2.imshow("Dilation",dilate)
    erosion = cv2.erode(dilate,kernel_3,iterations = 2)
    # cv2.imshow("Erosion",erosion)

    # contour detection-------------------------------------------------------------------------------------------
    erosion_copy=erosion.copy()
    contourimg = np.zeros((erosion_copy.shape[0], erosion_copy.shape[1], 3), dtype=np.uint8)
    contours, hierarchy = cv2.findContours(erosion_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contourimg, contours, -1, (0,0,255), 3)
    # cv2.imshow("Contour image",contourimg)

    hull=[]
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i],False))
    #create an empty black image
    drawing = np.zeros((th3.shape[0], th3.shape[1], 3), np.uint8) #blank black image

    for i in range(len(contours)): #drawing the convexhulls in the blank image
        cv2.drawContours(drawing,hull,i,(255,0,0),1,8)
    cv2.imshow("Convex hull",drawing)

    #draw shapes around the contours and track the contour
    for c in hull:
        contour_area=cv2.contourArea(c) #takes the contours and finds their area
        # print("Area = ",contour_area)
        if(contour_area>5000 ): #area condition
                M = cv2.moments(c) #used to calculate the center of mass/ centroid of the object/contour
                cx = int(M['m10']/M['m00'] )#formula
                cy = int(M['m01']/M['m00'])
                (x,y,w,h)= cv2.boundingRect(c) #enclose the contour in a rectangle , gives the x,y,width and height of the enclosing rectangle
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1) #to draw rectangle around the person
                # cv2.circle(img, (cx,cy), 1, (0, 0, 255), 3) #to draw circle at the centroid

    cv2.imshow("Output image",img)
    if cv2.waitKey(35) & 0xFF==ord('q'): #quit condition 
        break
camera.release() #stop output of video
cv2.destroyAllWindows() #close all imshow windows
