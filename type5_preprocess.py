import numpy as np
from PIL import Image

class processed_image():
    def __init__(self,path):
        self.img = np.array(Image.open(path).convert('L'), 'f')
        self.height,self.width = self.img.shape
        self.all_lines = [set() for i in range(self.width)]
        self.line = [-1] * self.width

    def show_image(self):
        img = self.img.astype('uint8')
        img = Image.fromarray(img)
        img.show()

    def get_all_lines(self):
        i = 0
        while i < self.height:
            if self.img[i][0] == 0:
                line_size = 0
                for j in range(5):
                    if self.img[i+j][0] != 0:
                        line_size = j

                        break

                self.line = [-1] * self.width
                self.line[0] = i
                for j in range(line_size):
                    self.all_lines[0].add(i+j)

                self.get_line(1,-1,line_size)
                i = i + line_size
            i = i + 1
        self.remove_line()
            # if self.img[i][self.width-1] == 0:
            #     self.line = [-1] * self.width
            #     self.line[self.width-1] = i
            #     self.get_line(self.width-2,1)
                # self.remove_line()

    def remove_line(self):
        for index, i in enumerate(self.all_lines):
            for j in i:
                self.img[j][index] = 255

    def get_line(self,n,num,line_size):
        if n < self.width:
            if self.line[n+num] > 0 and self.line[n+num] < self.height and self.img[self.line[n+num]][n] == 0 and self.img[self.line[n+num]+line_size-1][n] == 0:
                self.line[n] = self.line[n+num]
                # self.get_line(n-num,num,line_size)
            elif self.line[n+num] > 0 and self.line[n+num] < self.height and self.img[self.line[n+num]-1][n] == 0 and self.img[self.line[n+num]-1+line_size-1][n] == 0:
                self.line[n] = self.line[n+num] - 1
                # self.get_line(n-num,num,line_size)
            elif self.line[n+num] > 0 and self.line[n+num] < self.height and self.img[self.line[n+num]+1][n] == 0 and self.img[self.line[n+num]+1+line_size-1][n] == 0:
                self.line[n] = self.line[n+num] + 1
                # self.get_line(n-num,num,line_size)
            for i in range(line_size):
                self.all_lines[n].add(self.line[n]+i)
            self.get_line(n - num, num, line_size)



path = '/Users/lvyufeng/Documents/captcha_train_set/type5_train/type5_train_19924.png'

img = processed_image(path)
img.show_image()
img.get_all_lines()
img.show_image()