import csv
import numpy as np
from PIL import Image

def read_csv(path):
    train_data = []
    # valid_data = []
    csv_file = csv.reader(open(path,'r',encoding='utf-8'))
    for index,i in enumerate(csv_file):
        # if index <= 19500:
        #     train_data.append(i)
        # else:
        train_data.append(i[0])
    return train_data

path = '/Users/lvyufeng/Documents/captcha_train_set/type3_train/type3_train.csv'
train_data = read_csv(path)


t = np.zeros((50,250,3))

for i in train_data:

    path = '/Users/lvyufeng/Documents/captcha_train_set/type3_train/{}'.format(i)
    img = np.array(Image.open(path), 'f')
    img = img / 20000
    t = t + img
    pass

img = t.astype('uint8')
img = Image.fromarray(img)
# img.show()
img.save('type3_bg.jpg')

