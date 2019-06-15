import cv2
import numpy as np
from PIL import Image
imgname = '/Users/lvyufeng/Documents/captcha_train_set/type3_train/type3_train_1.jpg'

img = cv2.imread(imgname, cv2.IMREAD_COLOR)
bg = cv2.imread('model/type3_bg.png', cv2.IMREAD_COLOR)
img = img.astype(np.float32) - bg.astype(np.float32)
img[img < 0] = 0
img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
imt = np.asarray([0.3] * 60 + [0.2] * 65 + [0.08] * 125) * 255
im = (img > imt).astype(np.uint8) * 255
im0 = im.copy()
im = im[10:-5, :170]

img = im.astype('uint8')
img = Image.fromarray(img)
img.show()
# img.save('type2_processed_{}.jpg'.format(i))