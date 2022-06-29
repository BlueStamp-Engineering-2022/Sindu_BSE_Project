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