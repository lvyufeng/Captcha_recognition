import cv2
import numpy as np
from PIL import Image
imgname = '/Users/lvyufeng/Documents/captcha_train_set/type1_train/type1_train_1.jpg'
img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
im = 255 - img
retval, im = cv2.threshold(im, 255 * 0.3, 255, cv2.THRESH_BINARY)
lines = cv2.HoughLinesP(im, 1, 0.5 * np.pi / 180, 50, maxLineGap=5, minLineLength=30)
l = im - im
for i in range(lines.shape[0]):
    for x1, y1, x2, y2 in lines[i]:
        cv2.line(l, (x1, y1), (x2, y2), 255, 1)
im2 = im
im = im - l
im = 255 - im
img = im.astype('uint8')
img = Image.fromarray(img)
img.show()
img.save('hough_line.jpg')