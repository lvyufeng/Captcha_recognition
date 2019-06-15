import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image,ImageEnhance,ImageFilter

img_name = '/Users/lvyufeng/Documents/captcha_train_set/type1_train/type1_train_1.jpg'
#去除干扰线
imgname = '/Users/lvyufeng/Documents/captcha_train_set/type1_train/type1_train_1.jpg'
im = Image.open(img_name)
#图像二值化
enhancer = ImageEnhance.Contrast(im)
im = enhancer.enhance(2)
im = im.convert('1')
data = im.getdata()
w,h = im.size
# im.show()
im = Image.new('1', im.size, 'white')
black_point = 0
for x in range(1,w-1):
    for y in range(1,h-1):
        mid_pixel = data[w*y+x] #中央像素点像素值
        if mid_pixel == 0: #找出上下左右四个方向像素点像素值
            top_pixel = data[w*(y-1)+x]
            left_pixel = data[w*y+(x-1)]
            down_pixel = data[w*(y+1)+x]
            right_pixel = data[w*y+(x+1)]

            #判断上下左右的黑色像素点总个数
            if top_pixel == 0:
                black_point += 1
            if left_pixel == 0:
                black_point += 1
            if down_pixel == 0:
                black_point += 1
            if right_pixel == 0:
                black_point += 1
            if black_point >= 2:
                im.putpixel((x,y),0)
            #print black_point
            black_point = 0
# im.show()
im.save('cut_noisy_line.jpg')