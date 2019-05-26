from PIL import Image,ImageFilter,ImageEnhance
import numpy as np
import time
# import cv2
import itertools
import csv

path = '/home/ubuntu/dataset/type6_train/type6_train/{}'


index_list = [1,2,3,4,5,6,7,8,9]
# get_nine_char(path.format(3))
def get_permutations():
    perm_list = []
    for i in itertools.permutations(index_list, 4):
        t = ''.join([str(j) for j in i])
        # t = int(t)
        perm_list.append(t)
    return perm_list

perm_list = get_permutations()

def read_csv(path):
    csv_file = csv.reader(open(path,'r',encoding='utf-8'))
    for i in csv_file:
        print(i[-1])

def get_data(path,index,perm_list):
    # img = Image.open(path)
    img = np.array(Image.open(path).convert('L'), 'f')
    img[img >= 200] = 255
    img[img < 200] = 0

    x = [11, 78, 144] * 3
    y = [6, 6, 6, 46, 46, 46, 86, 86, 86]
    imgs = {}
    for i in range(9):
        imgs[str(i+1)] = img[y[i]:y[i] + 32, x[i]:x[i] + 32]

    labels = []
    pics = []
    for i in perm_list:
        matrix = [imgs[j] for j in i]
        pic = np.concatenate(matrix,axis=1)
        pics.append(pic)
        if i == index:
            labels.append(1)
            pic = pic.astype('uint8')
            pic = Image.fromarray(pic)
            pic.show()
        else:
            labels.append(0)

    img = img[125:157, 10:img.shape[1] - 15]
    img = img.astype('uint8')
    img = Image.fromarray(img)
    img.show()
    pass


height = 64
width = 240
mask = np.ones((height, width))
arr = np.zeros((height, width), np.float)
for i in range(1,20001):
    img = Image.open(path.format('type6_train_{}.png'.format(i))).convert('L')
    img = img.resize((width, height))
    imarr = np.array(img, dtype=np.float)
    imarr[imarr >= 210] = 255
    imarr[imarr < 210] = 0
    arr = arr + imarr / 20000
    # arr = np.array(np.round(arr), dtype=np.uint8)

out = arr.astype('uint8')
out[out >= 235] = 255
out[out < 235] = 0
out = Image.fromarray(out)  # save as gray scale
out.save('out.jpg')
out.show()
# get_data(path.format('type1_train_1.jpg'),'9713',perm_list)

# read_csv('/home/ubuntu/dataset/type1_train/type1_train.csv')
# get_permutations()
# for i in range(1,20000):
#     result = get_nine_char(path.format(i))
#     lens.add(result)
#
#
# # print(lens)
#     # key_v[len(result)] += 1
#     if i % 100 == 0:
#
#         print(lens)


