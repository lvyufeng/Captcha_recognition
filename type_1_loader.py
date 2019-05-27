import numpy as np
from preprocess import get_data,get_permutations,read_csv
import keras
import math
import itertools
from PIL import Image
class DataGenerator(keras.utils.Sequence):

    def __init__(self, datas, batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle
        self.path = '/Users/lvyufeng/Documents/captcha_train_set/type1_train/{}'
        self.index_list = [1,2,3,4,5,6,7,8,9]
    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas[k] for k in batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_datas)

        return X, y

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        cat_images = []
        images = []
        labels = []
        perm_list = self.get_permutations()
        # 生成数据
        for i, data in enumerate(batch_datas):
            #x_train数据
            cat_images_t, images_t, labels_t = self.get_data(self.path.format(data[0]),data[1],perm_list)
            cat_images.extend(cat_images_t)
            images.extend(images_t)
            labels.extend(labels_t)
        return [np.array(cat_images),np.array(images)], np.array(labels)

    def get_permutations(self):
        perm_list = []
        for i in itertools.permutations(self.index_list, 4):
            t = ''.join([str(j) for j in i])
            # t = int(t)
            perm_list.append(t)
        return perm_list

    def get_data(self,path, index, perm_list):
        # img = Image.open(path)
        img = np.array(Image.open(path).convert('L'), 'f')
        img[img >= 200] = 255
        img[img < 200] = 0

        x = [11, 78, 144] * 3
        y = [6, 6, 6, 46, 46, 46, 86, 86, 86]
        imgs = {}
        for i in range(9):
            imgs[str(i + 1)] = img[y[i]:y[i] + 32, x[i]:x[i] + 32]

        img = img[125:157, 10:img.shape[1] - 15]
        img = np.expand_dims(img,-1)
        cat_images = []
        images = []
        labels = []

        for i in perm_list:
            matrix = [imgs[j] for j in i]
            pic = np.concatenate(matrix, axis=1)
            pic = np.expand_dims(pic,-1)
            cat_images.append(pic)
            images.append(img)
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

        return cat_images,images, labels