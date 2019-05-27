from PIL import Image,ImageFilter,ImageEnhance
import numpy as np
import time
# import cv2
import itertools
import csv

path = '/Users/lvyufeng/Documents/captcha_train_set/type1_train/{}'
# path = '/home/ubuntu/dataset/type6_train/type6_train/{}'


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
    lines = []
    csv_file = csv.reader(open(path,'r',encoding='utf-8'))
    for i in csv_file:
        lines.append(i)
    return lines

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

    img = img[125:157, 10:img.shape[1] - 15]
    labels = []
    pics = []
    for i in perm_list:
        matrix = [imgs[j] for j in i]
        pic = np.concatenate(matrix,axis=1)
        pics.append((pic,img))
        if i == index:
            labels.append(1)
            # print(pic.shape)
            # pic = pic.astype('uint8')
            # pic = Image.fromarray(pic)
            # pic.show()
            # pic.save('cat.jpg')
        else:
            labels.append(0)

    # img = img[125:157, 10:img.shape[1] - 15]
    # imgs = [img] * len(pics)

    return pics,labels
    # print(img.shape)
    # img = img.astype('uint8')
    # img = Image.fromarray(img)
    # img.show()
    # img.save('img.jpg')
    # pass


# height = 64
# width = 240
# mask = np.ones((height, width))
# arr = np.zeros((height, width), np.float)
# for i in range(1,20001):
#     img = Image.open(path.format('5','5',i,'png')).convert('L')
#     img = img.resize((width, height))
#     imarr = np.array(img, dtype=np.float)
#     imarr[imarr >= 210] = 255
#     imarr[imarr < 210] = 0
#     arr = arr + imarr / 20000
#     # arr = np.array(np.round(arr), dtype=np.uint8)
#
# out = arr.astype('uint8')
# out[out >= 220] = 0
# out[out < 220] = 0
# h = np.sum(out,axis=0)
# # h = np.nonzero(h)[0]
# w = np.sum(out,axis=1)
# # w = np.nonzero(w)[0]
# pass
# out = Image.fromarray(out)  # save as gray scale
# out.save('out.jpg')
# out.show()
# get_data(path.format('1','1','1','jpg'),'9713',perm_list)

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


