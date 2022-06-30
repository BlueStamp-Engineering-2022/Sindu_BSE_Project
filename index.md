# Machine Learning for Hummingbird Detection Camera
This project uses image recognition software to capture hummingbirds. Artificial neural networks plus Python code are applied to hours of video footage to extract and concatenate clips containing hummingbirds coming to a sugar-water feeder. 

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Sindu Sirigineni | Evergreen Valley High School | Environmental/Software Engineering | Incoming Junior

![Headstone Image][![IMG-3396-1-1.jpg](https://i.postimg.cc/d10kF0bp/IMG-3396-1-1.jpg)]
  
# Final Milestone
My final milestone is the increased reliability and accuracy of my robot. I ameliorated the sagging and fixed the reliability of the finger. As discussed in my second milestone, the arm sags because of weight. I put in a block of wood at the base to hold up the upper arm; this has reverberating positive effects throughout the arm. I also realized that the forearm was getting disconnected from the elbow servo’s horn because of the weight stress on the joint. Now, I make sure to constantly tighten the screws at that joint. 

[![Final Milestone](https://res.cloudinary.com/marcomontalbano/image/upload/v1612573869/video_to_markdown/images/youtube--F7M7imOVGug-c05b58ac6eb4c4700831b2b3070cd403.jpg )](https://www.youtube.com/watch?v=F7M7imOVGug&feature=emb_logo "Final Milestone"){:target="_blank" rel="noopener"}

# Second Milestone
My final milestone is the increased reliability and accuracy of my robot. I ameliorated the sagging and fixed the reliability of the finger. As discussed in my second milestone, the arm sags because of weight. I put in a block of wood at the base to hold up the upper arm; this has reverberating positive effects throughout the arm. I also realized that the forearm was getting disconnected from the elbow servo’s horn because of the weight stress on the joint. Now, I make sure to constantly tighten the screws at that joint.

[![Third Milestone](https://res.cloudinary.com/marcomontalbano/image/upload/v1612574014/video_to_markdown/images/youtube--y3VAmNlER5Y-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=y3VAmNlER5Y&feature=emb_logo "Second Milestone"){:target="_blank" rel="noopener"}
# First Milestone
  

My first milestone was to complete the software portion of the project. I imported code from the original project and configured it for my project. The troubleshooting process took most of the second week and continued into the third. I imported a variety of tools to run my code, such as Tensorflow(an ML/AI library), ResNet(a residual learning network), the Matlab environment(to use a ground truth labeler), and Matplotlib(a Python plotting library), among other applications. I debugged the code and made sure it was free of errors, customized some code for ground truth labeling the images and generating plots, and also imported and debugged a separate code to split video footage into separate frames and save the frames as images. Overall, this is the purpose of the code: the video is turned into frames, where the ResNet network is applied to calculate the likelihood of a hummingbird being in the image. The image is cropped and resized, and the code also allows the ResNet to only run on specific sections of the image to make it more accurate. I created ground truth labels to improve the accuracy of the predictions, then finally I plotted the predictions that the images contained hummingbirds in a scatter plot.

## hummingbird.py
```markdown
### Code to detect whether an image contains any hummingbirds. It also includes an evaluation snippet.
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

### Subroutine to calculate likelihood for hummingbird in an image
def calculateLikelihood(x):
 x = np.expand_dims(x, axis=0)
 x = preprocess_input(x)
 preds = model.predict(x)
 # The class index for hummingbird is 94 in ImageNet.
 return preds[0][94]

### Load model and define parameters
model = ResNet50(weights='imagenet')
dim = 861
height = 1098
width = 2234
expecteddim = 224
data = []
N = 12
### Loop through the images
for n in range(1, N):
  # Read the image
  name = 'thumb' + '{0:04d}'.format(n) + '.jpeg'
  img_path = name
  img = image.load_img(img_path)
  # Crop the image, resize it and make prediction
  startx = 0
  starty = 0
  endx = startx + dim
  endy = starty + dim
  cropped_img = img.crop((startx, starty, endx, endy))
  resize_img = cropped_img.resize((expecteddim, expecteddim), Image.ANTIALIAS)
  x = image.img_to_array(resize_img)
  p1 = calculateLikelihood(x)
  # Repeat above but close to another flower
  startx = width - dim
  starty = 0
  endx = startx + dim
  endy = starty + dim
  cropped_img = img.crop((startx, starty, endx, endy))
  resize_img = cropped_img.resize((expecteddim, expecteddim), Image.ANTIALIAS)
  x = image.img_to_array(resize_img)
  p2 = calculateLikelihood(x)
  # Take the maximum likelihood as the return likelihood
  print(n, max(p1, p2))
  data.append(max(p1, p2))

### Compare with ground truth labels
trueXList = []
trueYList = []
falseXList = []
falseYList = []
ground_truth = {"thumb0001.jpeg": ["t", 0] , "thumb_0002.jpeg": ["f", 1], "thumb0003.jpeg": ["t",2], "thumb0004.jpeg": ["t", 3], "thumb0005.jpeg": ["t", 4], "thumb0006.jpeg": ["t",5], "thumb0007.jpeg": ["t",6] , "thumb0008.jpeg": ["t", 7], "thumb0009.jpeg": ["t", 8], "thumb0010.jpeg": ["t", 9], "thumb0011.jpeg": ["t",10]}
for img in ground_truth:
	img_info = (ground_truth[img])
	index = img_info[1]
	print(img_info[0])
	if img_info[0] == "t":
		trueXList.append(img_info[1])
		trueYList.append(data[index - 1])
	else:
		falseXList.append(img_info[1])
		falseYList.append(data[index - 1])

### Generate plots
plt.scatter(falseXList, falseYList, color='red')
plt.scatter(trueXList, trueYList, color='green')
plt.show()
```
## pics.py
```markdown
### Code to split video footage into frames and save the separate frames as images.
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
```
[![First Milestone](https://res.cloudinary.com/marcomontalbano/image/upload/v1612574117/video_to_markdown/images/youtube--CaCazFBhYKs-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=CaCazFBhYKs "First Milestone"){:target="_blank" rel="noopener"}

# Starter Project
  

My starter project is the customizable Arduino project, in which I used the Piezo buzzer as my output. The Piezo buzzer plays various tones at different frequencies to create a song. The materials I used are: an Arduino Uno microboard, an Arduino Proto Shield, two jumper wires, and a Piezo buzzer. First, I soldered on three headers to the Proto Shield. Using these headers, I was able to connect the Proto Shield and the Uno. Now that the two components were connected, I built my circuit: I connected one jumper wire from the positive side of the Piezo buzzer to pin 9 on the Proto Shield, and another wire from the negative side to the ground pin(GND). I then connected the Arduino Uno to my laptop using a USB cable. The project came with Arduino code, which: corresponded frequencies to note characters, defined the song length, defined the note sequence and their corresponding beats, the tempo, and included other loops and functions to play the song. I imported it and ran the code, and the buzzer played a simple melody. Afterwards, I experimented with changing the code to play different songs by tweaking the note sequence, song length, and beats.


[![First Milestone](http://i3.ytimg.com/vi/wGRu2dGmE5Q/hqdefault.jpg)](https://www.youtube.com/watch?v=wGRu2dGmE5Q "First Milestone"){:target="_blank" rel="noopener"}
