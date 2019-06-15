import cv2
import numpy as np
from PIL import Image
imgname = '/Users/lvyufeng/Documents/captcha_train_set/type2_train/type2_train_1.jpg'
img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
I0 = cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, np.ones((5, 5), dtype='uint8'))
I1 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, np.ones((5, 5), dtype='uint8'))
img_closed = cv2.add(I0, I1)

retval, t = cv2.threshold(img_closed, 125, 1, cv2.THRESH_BINARY)
h_sum = t.sum(axis=0)
v_sum = t.sum(axis=1)
x1, x2 = (v_sum > 1).nonzero()[0][0], (v_sum > 1).nonzero()[0][-1]
y1, y2 = (h_sum > 5).nonzero()[0][0], (h_sum > 1).nonzero()[0][-1]
im = img[x1:x2, y1:y2]


imgs = np.zeros((5, 1, 32, 32), dtype=np.uint8)
t = im.shape[1] / 5.0
dd = 4
bb = np.zeros((im.shape[0], dd), dtype=np.uint8) + 255
im1 = im.transpose()[0:int(np.floor(t)) + dd].transpose()
im2 = im.transpose()[int(np.floor(t)) - dd:int(np.floor(2 * t)) + dd].transpose()
im3 = im.transpose()[int(np.floor(2 * t)) - dd:int(np.floor(3 * t)) + dd].transpose()
im4 = im.transpose()[int(np.floor(3 * t)) - dd:int(np.floor(4 * t)) + dd].transpose()
im5 = im.transpose()[int(np.floor(4 * t)) - dd:].transpose()
imgs[0, 0] = cv2.resize(np.concatenate((bb, im1), axis=1), (32, 32))
imgs[1, 0] = cv2.resize(im2, (32, 32))
imgs[2, 0] = cv2.resize(im3, (32, 32))
imgs[3, 0] = cv2.resize(im4, (32, 32))
imgs[4, 0] = cv2.resize(np.concatenate((im5, bb), axis=1), (32, 32))

# imgs = imgs.astype('float32') / 255.0
for i in range(5):
    img = imgs[i, 0].astype('uint8')
    img = Image.fromarray(img)
    # img.show()
    img.save('type2_processed_{}.jpg'.format(i))
