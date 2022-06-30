
import cv2
import os

# Opens the inbuilt camera of laptop to capture video.
cam = cv2.VideoCapture("VID_20220401_114955_LS.mp4")
#cam = cv2.VideoCapture("C:\\Users\\school\\Desktop\\summer\\bse\\VID_20220401_114955_LS.mp4")

try:
      
    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')
  
# frame
currentframe = 0
  
while(True):
      
    # reading from frame
    ret,frame = cam.read()
  
    if ret:
        # if video is still left continue creating images
        name = './data/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)
  
        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break
  
cam.release()
#cv2.destroyAllWindows()