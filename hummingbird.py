### Code to detect whether an image contains any hummingbirds. It also includes an evaluation snippet.
import ssl
# (1)
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np # NumPy is a numerical mathematics extension; also is a Python library, used for working with arrays
import matplotlib.pyplot as plt # Matplotlib is a plotting library for Python and NumPy (for data visualization and graphical plotting)
from PIL import Image # PIL: Python Imaging Library (support for opening, manipulating, and saving many different image file formats(image editing capabilities))

# (2)
### Subroutine(or rather a function since it returns a value) to calculate likelihood for hummingbird in an image; subroutines vs functions: used interchangeably since the syntax is the same
def calculateLikelihood(x): # x is the placeholder of a value that you pass to the function from where it is called
 x = np.expand_dims(x, axis=0) # remember we imported NumPy as np; expand_dims() function used to expand the shape of an array (the dimensions)
 x = preprocess_input(x) # meant to adequate your image to the format the model requires (resizes it)
 preds = model.predict(x) # preds is the name of the variable (variable can have any name); predict() function makes use of the learned label to map and predict the labels for the data to be tested
 # The class index for hummingbird is 94 in ImageNet.
 return preds[0][94]

# (4)
### Load model and define parameters
model = ResNet50(weights='imagenet') # weights is an optional parameter which is used to weigh the possibility for each value
dim = 861 # dim is just a variable i guess?
height = 1098
width = 2234
expecteddim = 224
data = []
N = 12
### Loop through the images
for n in range(1, N):
  # Read the image
  name = 'thumb' + '{0:04d}'.format(n) + '.jpeg' # this line creates image files (?) ; {0:04d} adds 4 decimal places, d indicates a digit(0-9); .format(n) formats the string; (ex: the image files will look like thumb0001.jpg, thumb0002.jpg, etc)
  img_path = name
  img = image.load_img(img_path)
  # Crop the image, resize it and make prediction
  startx = 0
  starty = 0
  endx = startx + dim
  endy = starty + dim
  cropped_img = img.crop((startx, starty, endx, endy))
  resize_img = cropped_img.resize((expecteddim, expecteddim), Image.ANTIALIAS) # antialiasing is a technique used in digital imaging to reduce the visual defects that occur when high-resolution images are presented in a lower resolution (smooths the image/edges)
  x = image.img_to_array(resize_img) # img_to_array : converts a PIL Image instance to a Numpy array
  p1 = calculateLikelihood(x) # * * what is p1?? * *
  # Repeat above but close to another flower
  startx = width - dim
  starty = 0
  endx = startx + dim
  endy = starty + dim
  cropped_img = img.crop((startx, starty, endx, endy))
  resize_img = cropped_img.resize((expecteddim, expecteddim), Image.ANTIALIAS)
  x = image.img_to_array(resize_img) #img_to_array converts a PIL Image instance to a Numpy array
  p2 = calculateLikelihood(x)
  # Take the maximum likelihood as the return likelihood
  print(n, max(p1, p2))
  data.append(max(p1, p2)) # data.append: process that involves adding new data elements to an existing database

# (5)
### Compare with ground truth labels (ground truth: information that is known to be real or true)
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
		#print("TEST")
		trueXList.append(img_info[1])
		trueYList.append(data[index - 1])
	else:
		falseXList.append(img_info[1])
		falseYList.append(data[index - 1])


'''
f = open('gt.txt', 'r', errors='replace') #f means formatted string literals(f-string): these strings may contain replacement fields as opposed to other string literals, which always have a constant value; 'r' is for reading the file
tmp = f.readline().rstrip() # tmp = tempfile: a Python module used in a situation, where we need to read multiple files, change or access the data in the file, and gives output files based on the result of processed data
for i in range(1, N):
  info = tmp.split() # split function
  index = int(info[2]) # int() function converts the specified value into an integer number
  if info[1] == 't': # the == operator compares the value or quality of two objects
    trueXList.append(index)
    trueYList.append(data[index - 1])
  else:
    falseXList.append(index)
    falseYList.append(data[index - 1])
  tmp = f.readline().rstrip() #the readline() method returns one line from the file; the rstrip() method removes any trailing characters (characters at the end a string)
f.close() # closes the opened file
'''
# (6)
### Generate plots
plt.scatter(falseXList, falseYList, color='red') #graphs scatter plot
plt.scatter(trueXList, trueYList, color='green')
# plt.hlines(y=0.05, xmin=0, xmax=N, color='b') # plots horizontal lines in a graph at each y from xmin to xmax
plt.show() # starts an  event loop, looks for all currently active figure objects, and opens one or more interactive windows that display your figure or figures