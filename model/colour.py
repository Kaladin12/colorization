import cv2 as cv
import matplotlib.pyplot as plt
from time import sleep

image = cv.imread(r"F:\MEGA\CETYS\sechs\vision_artificial\archive\data\train_black\all\image0000.jpg")
plt.imshow(image, cmap = 'gray')
sleep(5)

