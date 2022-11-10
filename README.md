# Coursework PART I - Shape Detection

> This part of the coursework requires [Python 3.6](https://www.python.org/downloads/).
> For Windows, you might want to use [Conda](https://www.anaconda.com/products/distribution). 
 
## Introduction
Detecting (locating & classifying) instances of an object class in images is an important application in computer vision as well as an ongoing area of research. This assignment requires you 1) to experiment with the classical Viola-Jones object detection framework as discussed in the lectures and provided by OpenCV, and 2) to combine it with other detection approaches to improve its efficacy. Your approach is to be tested and evaluated on a small image set that depicts aspects of an important traffic sign – No Entry.

## Subtask 1: The No-Entry Sign Detector
_(15 marks)_

This subtask requires you to build an object detector that recognises no entry signs. The initial steps of this subtask introduce you to OpenCV’s boosting tool, which you can use to construct an object detector that utilises Haar-like features. 

1. Create virtual environment with Python 3.6 `conda create -n ipcv36 python=3.6`, activate your environment `conda activate ipcv36`, and install OpenCV packages `conda install -c menpo opencv`. Check OpenCV verion with `python -c 'import cv2; print(cv2.__version__)'`. It should be 3.4.x.
2. You are given `no_entry.jpg` containing a no entry sign that can serve as a prototype for generating a whole set of positive training images. 
3. Unzip `negatives.zip` and keep all negative images in a directory called `negatives`. A text file `negatives.dat` lists all filenames in the directory.
4. Create your positive training data set of 500 samples of no entry signs from the single prototype image provided. To do this, you can run the tool `opencv_createsamples` via the following single command and execute it in a folder that contains the negatives.dat file, the no_entry.jpg image and the negatives folder: 

```
opencv_createsamples -img no_entry.jpg -vec no_entry.vec  -w 20 -h 20 -num 500 -maxidev 80 -maxxangle 0.8 -maxyangle 0.8 -maxzangle 0.2
```
5. This will create 500 tiny 20×20 images of no entry signs (later used as positive training samples) and store in the file `no_entry.vec`, which contains all these 500 small sample images. Each of the sample images is created by randomly changing viewing angle and contrast (up to the maximum values specified) to reflect the possible variability of viewing parameters in real images better.
6. Now use the created positive image set to train a no entry sign detector via AdaBoost. To do this, create a directory called `NoEntrycascade` in your working directory. Then run the `opencv_traincascade` tool with the following parameters as a single command in your working directory:
```
opencv_traincascade -data NoEntrycascade -vec no_entry.vec -bg negatives.dat -numPos 500 -numNeg 500 -numStages 3 -maxDepth 1 -w 20 -h 20 -minHitRate 0.999  -maxFalseAlarmRate 0.05 -mode ALL
```
7. This will start the boosting procedure and construct a strong classifier stored in the file cascade.xml, which you can load in an OpenCV program for later detection as done in Lab4: Face Detection (`face.py`). You might need the change `model = cv2.CascadeClassifier()` to `model = cv2.CascadeClassifier(cascade_name)` or remove `cv2.samples.findFile`.
8. During boosting the tool will provide updates about the machine learning in progress. Here is an example output when using 1000 instead of 500 samples…
9. The boosting procedure considers all the positive images and employs sampled patches from the negative images to learn. The detector window will be 20×20. To speed up the detection process, the strong classifier is built in 3 parts (numStages) to form an attentional cascade as discussed in the Viola-Jones paper. The training procedure may take up to 15min for reasons discussed in the lectures – stop the training and restart if it exceeds this time. 




