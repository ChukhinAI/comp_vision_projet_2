import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import numpy
# load the trained model to classify sign
from keras.models import load_model

#  preprocessing
import time  # for measure time of procedure execution
import numpy as np
import math
import cv2
import imutils
import skimage.transform
import matplotlib.cbook as cbook

# =================================================================================================
#                                       preprocessing
# =================================================================================================


def FindImages(pic):
    """
    find traffic sign
    """
    # array of resized 32x32x3 images finded during search procedure
    images = []

    """ first step - preparing picture"""
    # read the picture
    image = cv2.imread(pic)
    # cv2_imshow(image)
    # define dimensions of image and center
    # measurement of picture starts from top left corner
    height, width = image.shape[:2]
    # print(str(height)+" "+str(width))
    center_y = int(height / 2)
    center_x = int(width / 2)

    # define array of distance from center of image - it connected with area of contour
    # more distance from center - more bigger contour (look at the picture
    # test_1.jpg - 3 red squares shows this areas)
    dist_ = [center_x / 3, center_x / 2, center_x / 1.5]

    # defining main interest zone of picture (left, right, top bottom borders)
    # this zone is approximate location of traffic sings
    # green zone at the picture test_1.jpg
    left_x = center_x - int(center_x * .7)
    right_x = width
    top_y = 0
    bottom_y = center_y + int(center_y * .3)
    # crop zone of traffic signs location to search only in it
    crop_image = image[top_y:bottom_y, left_x:right_x]
    # cv2_imshow(crop_image)

    # make canny image - first image for recognition of shapes
    # look at the test_1_crop_canny.jpg
    canny = cv2.Canny(crop_image, 50, 240)
    blur_canny = cv2.blur(canny, (2, 2))
    _, thresh_canny = cv2.threshold(blur_canny, 127, 255, cv2.THRESH_BINARY)

    # make color HSV image - second image for color mask recognition
    # Convert BGR to HSV
    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)

    # define the list of boundaries (lower and upper color space for
    # HSV
    # mask for red color consist from 2 parts (lower mask and upper mask)
    # lower mask (0-10)
    lower_red = np.array([0, 50, 50], np.uint8)
    upper_red = np.array([10, 255, 255], np.uint8)
    mask_red_lo = cv2.inRange(hsv, lower_red, upper_red)
    # upper mask (170-180)
    lower_red = np.array([160, 50, 50], np.uint8)
    upper_red = np.array([180, 255, 255], np.uint8)
    mask_red_hi = cv2.inRange(hsv, lower_red, upper_red)
    # blue color mask
    lower_blue = np.array([100, 50, 50], np.uint8)
    upper_blue = np.array([140, 200, 200], np.uint8)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # yellow color mask
    lower_yellow = np.array([15, 110, 110], np.uint8)
    upper_yellow = np.array([25, 255, 255], np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # join all masks
    # could be better to join yellow and red mask first  - it can helps to detect
    # autumn trees and delete some amount of garbage, but this is TODO next
    mask = mask_red_lo + mask_red_hi + mask_yellow + mask_blue

    # find the colors within the specified boundaries and apply
    # the mask
    hsv_out = cv2.bitwise_and(hsv, hsv, mask=mask)

    # encrease brightness TODO later
    # h, s, v = cv2.split(hsv_out)
    # v += 50
    # bright_hsv_out = cv2.merge((h, s, v))

    # blurred image make lines from points and parts and increase quality (1-3,1-3) points
    blur_hsv_out = cv2.blur(hsv_out, (1, 1))  # change from 1-3 to understand how it works

    # preparing HSV for countours - make gray and thresh
    gray = cv2.cvtColor(blur_hsv_out, cv2.COLOR_BGR2GRAY)
    # increasing intensity of finded colors with 0-255 value of threshold
    # look at the file test_1_hsv_binary to understand what the file thresh is
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # do not need to mix the file - it will be problem with contour recognition
    # dst = cv2.addWeighted(canny,0.3,thresh,0.7,0)
    # cv2.imshow('img1',thresh_canny)
    # cv2.imshow('img2',thresh)
    # cv2.waitKey(0)

    """step two - searching for contours in prepared images"""
    # calculating of finded candidates
    multiangles_n = 0

    # contours of the first image (thresh_canny)
    # cv2.RETR_TREE parameter shows all the contours internal and external
    contours1, _ = cv2.findContours(thresh_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("Contours total at first image: "+str(len(contours1)))

    # take only first  biggest 15% of all elements
    # skipping small contours from tree branches etc.
    contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)[:int(len(contours1) / 6)]
    for cnt in contours1:
        # find perimeters of area - if it small and not convex - skipping
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 25 or cv2.isContourConvex == 'False':  # 25 - lower - more objects higher-less
            continue

        # calculating rectangle parameters of contour
        (x, y), (w, h), angle = cv2.minAreaRect(cnt)
        # calculating koefficient between width and height to understand if shape is looks like traffic sign or not
        koeff_p = 0
        if w >= h and h != 0:
            koeff_p = w / h
        elif w != 0:
            koeff_p = h / w
        if koeff_p > 2:  # if rectangle is very thin then skip this contour
            continue

            # compute the center of the contour
        M = cv2.moments(cnt)
        cX = 0
        cY = 0
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        # transform cropped image coordinates to real image coordinates
        cX += left_x
        cY += top_y

        dist_c_p = math.sqrt(math.pow((center_x - cX), 2) + math.pow((center_y - cY), 2))
        # skipping small contours close to the left and right sides of picture
        # remember res squares from test_1.jpg files? :)
        if dist_c_p > dist_[0] and dist_c_p <= dist_[1] and perimeter < 30:
            continue
        if dist_c_p > dist_[1] and dist_c_p <= dist_[2] and perimeter < 50:
            continue
        if dist_c_p > dist_[2] and perimeter < 70:
            continue
        # 0,15 - try to use different koefficient to better results
        approx_c = cv2.approxPolyDP(cnt, 0.15 * cv2.arcLength(cnt, True),
                                    True)  # 0,15 - lower - more objects higher-less
        if len(approx_c) >= 3:  # if contour has more then two angles...
            # calculating parameters of rectangle around contour to crop ROI of porential traffic sign
            x, y, w_b_rect, h_b_rect = cv2.boundingRect(cnt)
            # cv2.rectangle(image,(cX-int(w_b_rect/2)-10,cY-int(h_b_rect/2)-10),(cX+int(w_b_rect/2)+10,cY+int(h_b_rect/2)+10),(255,0,0),1)
            # put this ROI to images array for next recognition
            images.append(image[cY - int(h_b_rect / 2) - 3:cY + int(h_b_rect / 2) + 3,
                          cX - int(w_b_rect / 2) - 3:cX + int(w_b_rect / 2) + 3])
            # save to the file - will be skip later TODO
            # cv2.imwrite("%recogn.jpg" % multiangles_n,image[cY-int(h_b_rect/2)-3:cY+int(h_b_rect/2)+3, cX-int(w_b_rect/2)-3:cX+int(w_b_rect/2)+3])
            # increasing multiangles quantity
            multiangles_n += 1

    # contours in second image (thresh)
    # in this picture we are only use RETR_EXTERNAL contours to avoid processing for example windows in yellow and red houses
    # and holes between plants etc
    contours2, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("Contours total at second image: "+str(len(contours2)))

    # make first 10% biggest contours +- of elements
    contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:int(len(contours2) / 10)]

    for cnt in contours2:
        # calculating perimeter
        perimeter = cv2.arcLength(cnt, True)
        # if perimeter id too big or too small and is not convex skipping
        if perimeter > 200 or perimeter < 25 or cv2.isContourConvex == 'False':  # 25 - lower - more objects higher-less
            continue

        # calculating rectangle parameters of contour
        (x, y), (w, h), angle = cv2.minAreaRect(cnt)
        # calculating koefficient between width and height to understand if shape is looks like traffic sign or not
        koeff_p = 0
        if w >= h and h != 0:
            koeff_p = w / h
        elif w != 0:
            koeff_p = h / w
        if koeff_p > 2:  # 1.3, if rectangle is very thin then skip this contour
            continue

        # compute the center of the contour
        M = cv2.moments(cnt)
        cX = 0
        cY = 0
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        # transform cropped image coordinates to real image coordinates
        cX += left_x
        cY += top_y

        dist_c_p = math.sqrt(math.pow((center_x - cX), 2) + math.pow((center_y - cY), 2))
        # skipping small contours close to the left and right sides of picture
        if dist_c_p > dist_[0] and dist_c_p <= dist_[1] and perimeter < 30:
            continue
        if dist_c_p > dist_[1] and dist_c_p <= dist_[2] and perimeter < 50:
            continue
        if dist_c_p > dist_[2] and perimeter < 70:
            continue

        approx_c = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True),
                                    True)  # 0,03 - lower - more objects higher-less
        if len(approx_c) >= 0:  # 3 -> 2
            x, y, w_b_rect, h_b_rect = cv2.boundingRect(cnt)
            # cv2.rectangle(image,(cX-int(w_b_rect/2)-10,cY-int(h_b_rect/2)-10),(cX+int(w_b_rect/2)+10,cY+int(h_b_rect/2)+10),(0,255,0),1)
            curImage = image[cY - int(h_b_rect / 2) - 3:cY + int(h_b_rect / 2) + 3,
                       cX - int(w_b_rect / 2) - 3:cX + int(w_b_rect / 2) + 3]
            if curImage.size != 0:
                images.append(image[cY - int(h_b_rect / 2) - 3:cY + int(h_b_rect / 2) + 3,
                              cX - int(w_b_rect / 2) - 3:cX + int(w_b_rect / 2) + 3])
                # cv2.imwrite("extr/%recogn.jpg" % multiangles_n, curImage)
                # print('s = ', multiangles_n)
                # multiangles_n += 1

    # print(str(multiangles_n) + ' showed multiangles')

    # cv2.imshow('img',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    i = len(images)
    return images, i


# main program

np.seterr(divide='ignore', invalid='ignore')
start_time = time.time()
# image_set = FindImages('tests/25.jpg') # original
# print(image_set)
# Resize images
# images32 = [skimage.transform.resize(image, (32, 32)) for image in image_set]
print("--- %s seconds ---" % (time.time() - start_time))
numb = 0
'''
image_set = FindImages(file_path)
for img in image_set:
    #print('shape = ', len(image_set))
    #cv2.imshow((str)(numb), img)
    cv2.imwrite("extr/%recogn.jpg" % numb, img)
    numb += 1
    cv2.waitKey()
'''

# =================================================================================================
#                                       detection
# =================================================================================================


model1 = load_model('models\\0_traffic_classifier.h5')  # original # 4 неверно | 2 | 3 | 1 | 3 |
model3 = load_model('models\\1-30_traffic-signs.h5')  # 1-30 GerTSc
model4 = load_model('models\\2-33_TSD.h5')  # 2-33_TSD
model2 = load_model('models\\6-10_model.h5')  # 5-8_1 # 7-38_1_sign_classifier # 4 неверно | 3 | 1 | 3 | 1
# new
# model = load_model('models\\10-model-31x31.h5')  # более 8 ложных, 1 раз не узнал 30 кмч

# dictionary to label all traffic signs class. # original
classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           # 7: 'End of speed limit (80km/h)',
           7: 'Cannot recognize',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'Cannot recognize'}
'''
           #10:'No passing',   
           #11:'No passing veh over 3.5 tons',     
           #12:'Right-of-way at intersection',     
           #3:'Priority road',    
           #4:'Yield',     
           #15:'Stop',       
           #6:'No vehicles',       
           #7:'Veh > 3.5 tons prohibited',       
           #8:'No entry',       
           #19:'General caution',     
           #20:'Dangerous curve left',      
           #21:'Dangerous curve right',   
           #22:'Double curve',      
           #23:'Bumpy road',     
           #24:'Slippery road',       
           #25:'Road narrows on the right',  
           #26:'Road work',    
           #27:'Traffic signals',      
           #28:'Pedestrians',     
           #29:'Children crossing',     
           #30:'Bicycles crossing',       
           #31:'Beware of ice/snow',
           #32:'Wild animals crossing',      
           #33:'End speed + passing limits',      
           #34:'Turn right ahead',     
           #35:'Turn left ahead',       
           #36:'Ahead only',      
           #37:'Go straight or right',      
           #38:'Go straight or left',      
           #39:'Keep right',     
           #40:'Keep left',      
           #41:'Roundabout mandatory',     
           #42:'End of no passing',      
           #43:'End no passing veh > 3.5 tons'} 
            '''


def classify(file_path):
    global label_packed
    # image = Image.open(file_path)
    img_read = plt.imread(file_path)

    image = Image.open(file_path)
    image1 = image.resize((30, 30))
    image1 = numpy.expand_dims(image1, axis=0)
    image1 = numpy.array(image1)
    pred1 = model1.predict_classes([image1])[0]

    image = Image.open(file_path)
    image2 = image.resize((48, 48))
    image2 = numpy.expand_dims(image2, axis=0)
    image2 = numpy.array(image2)
    pred2 = model2.predict_classes([image2])[0]

    image = Image.open(file_path)
    image3 = image.resize((32, 32))
    image3 = numpy.expand_dims(image3, axis=0)
    image3 = numpy.array(image3)
    pred3 = model3.predict_classes([image3])[0]

    image = Image.open(file_path)
    image4 = image.resize((32, 32))
    image4 = numpy.expand_dims(image4, axis=0)
    image4 = numpy.array(image4)
    pred4 = model4.predict_classes([image4])[0]

    if pred1 > 8:
        pred1 = 9

    if pred2 > 8:
        pred2 = 9

    if pred3 > 8:
        pred3 = 9

    if pred4 > 8:
        pred4 = 9

    sign = classes[pred4+1]
    print('m_4_32_', sign)

    sign = classes[pred3+1]
    print('m_3_32', sign)

    sign = classes[pred2+1]
    print('m_2_48', sign)

    sign = classes[pred1+1]
    print('m_1_30', sign)
    print('-------------------------')

    if (pred1 < 9) and (pred1 == pred2):
        if ((pred4 < 9) and (pred4 != 7)) or ((pred3 < 9) and (pred3 != 7)):
            plt.title(sign)
            plt.imshow(img_read)
            plt.show()


file_path = 'signs/56.jpg'
pred_imgs, numb_susp = FindImages(file_path)
print('numb_susp = ', numb_susp)
numb = 0
# print('shape = ', len(pred_imgs))
for img in pred_imgs:
    # cv2.imshow((str)(numb), img)
    #print('type = ', img.shape)
    # classify(img)
    cv2.imwrite("signs/extr/%recogn.jpg" % numb, img)
    classify("signs/extr/%recogn.jpg" % numb)
    numb += 1
    # cv2.waitKey()
