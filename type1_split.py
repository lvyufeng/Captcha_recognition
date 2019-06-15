import numpy as np
from PIL import Image
path = '/Users/lvyufeng/Documents/captcha_train_set/type1_train/type1_train_1.jpg'
img = np.array(Image.open(path).convert('L'), 'f')
img[img >= 200] = 255
img[img < 200] = 0

x = [11, 78, 144] * 3
y = [6, 6, 6, 46, 46, 46, 86, 86, 86]
imgs = {}
# for i in range(9):
#     image = img[y[i]:y[i] + 32, x[i]:x[i] + 32]
#     image = image.astype('uint8')
#     image = Image.fromarray(image)
#     # image.show()
#     image.save('type1_train_1_{}.jpg'.format(i))

img = img[125:157, 10:img.shape[1] - 70]
for i in range(4):
    image = img[:,i*int(img.shape[1]/4):(i+1)*int(img.shape[1]/4)]
    image = image.astype('uint8')
    image = Image.fromarray(image)
    image.show()
    image.save('type1_train_1_code_{}.jpg'.format(i))