from PIL import Image,ImageFilter,ImageEnhance
import numpy as np
import time
import cv2
path = '/Users/lvyufeng/Documents/captcha_train_set/type1_train/type1_train_{}.jpg'


def simple_cut(vert):
    min_thresh = 1  # 字符上最少的像素点
    min_range = 25  # 字符最小的宽度
    begin, end = 0, 0
    cuts = []
    for i, count in enumerate(vert):
        if count >= min_thresh and begin == 0:
            begin = i
        elif count >= min_thresh and begin != 0:
            continue
        elif count <= min_thresh and begin != 0:
            end = i
            # print (begin, end), count
            if end - begin >= min_range:
                cuts.append((begin, end))

                begin = 0
                end = 0
                continue
        elif count <= min_thresh or begin == 0:
            continue
    return cuts


def get_nine_char(image):

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    im = 255 - img
    retval, im = cv2.threshold(im, 255 * 0.3, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(im, 1, 0.5 * np.pi / 180, 50, maxLineGap=5, minLineLength=30)
    l = im - im
    for i in range(lines.shape[0]):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(l, (x1, y1), (x2, y2), 255, 1)
    im2 = im
    im = im - l
    x = [11, 78, 144] * 3
    y = [6, 6, 6, 46, 46, 46, 86, 86, 86]
    imgs = np.zeros((9, 1, 32, 32), np.float32)
    for i in range(9):
        imgs[i] = im2[y[i]:y[i] + 32, x[i]:x[i] + 32]


    img = im2[125:157, 10:im.shape[1] - 15]
    t = im[125:157, 10:im.shape[1] - 15]
    t = cv2.medianBlur(t, 3)
    d = Image.fromarray(t)
    d.show()
    retval, t = cv2.threshold(t, 255 * 0.5, 1, cv2.THRESH_BINARY)

    s = np.sum(t[5:, :], axis=0)
    s2 = np.sum(t[3:7, :], axis=0)
    results = simple_cut(s)
    results_e = []
    for i in results:
        if np.sum(s2[i[0]:i[1]]) != 0:
            results_e.append(i)
    # s2 = np.sum(t[0:7, :], axis=0)
    # s[s <= 4] = 0
    # s2[s2 > 0] = 1
    # p = s.nonzero()
    # img = img[:, 0:max(110, min(s.nonzero()[0][-1] + 5, s.shape[0]))]
    # print(img.shape[1])
    # idxl = list(range(0, img.shape[1] - 32, 30))
    # idxr = [i for i in map(lambda x: img.shape[1] - x - 32, idxl)]
    # idxl = idxl[:3]
    # idxr = idxr[:3]
    # idxs = idxl + idxr
    # idxs.sort()
    # imgs = np.zeros((len(idxs), 1, 32, 32), np.float32)
    # result = []
    # for i in range(len(idxs)):
    #     if i > 1 and idxs[i] - idxs[i-1] < 10:
    #         continue
    #     # imgs[i] = img[0:32, idxs[i]:idxs[i] + 32]
    #     # print(np.sum(t[7][idxs[i]:idxs[i] + 32]))
    #     if np.sum(s2[idxs[i]+8:idxs[i] + 24]) != 0 :
    #         result.append(idxs[i])
    #     # else:
    #
    #     #     t = img[0:32, idxs[i]:idxs[i] + 32]
    #     #     t = Image.fromarray(t)
    #     #     t.show()
    #     # time.sleep(1)
    # # img = Image.fromarray(img[1:,:w-1])
    # # img.show()
    # # if len(result) != 3:
    # #     d = Image.fromarray(t)
    # #     d.show()
    return img.shape[1]

# key_v = {
#     0:0,
#     1:0,
#     2:0,
#     3:0,
#     4:0,
#     5:0,
#     6:0
# }
# lens = set()
get_nine_char(path.format(3))
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


