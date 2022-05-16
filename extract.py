import PIL.Image as Image, os, sys, array, math
import numpy as np
import math


def getBytes(file_path):
    fileobj = open(file_path, mode='rb')
    buffer = array.array('B', fileobj.read())  #二进制(8个一)占大小为一个字节，换成十进制变成255
    size = len(buffer)
    #print(size / 1024, 'kB')
    fileobj.close()
    return buffer, size


def _calImgWidth(size):
    size /= 1024
    width = 512
    if size < 10:
        width = 16
    elif size < 30:
        width = 32
    elif size < 30:
        width = 32
    elif size < 60:
        width = 64
    elif size < 100:
        width = 128
    elif size < 200:
        width = 192
    elif size < 500:
        width = 256
    elif size < 1000:
        width = 384
    return width


def getBytesMatrix(buffer, buffer_size):
    width = _calImgWidth(buffer_size)
    height = math.ceil(buffer_size / width / 3)
    img = np.pad(buffer, (0, width * height * 3 - buffer_size),
                 'constant').reshape((height, width, 3))
    return Image.fromarray(img.astype('uint8')).convert('RGB')


def getLocMatrix(buffer, buffer_size):
    b = np.pad(buffer, (0, 0 if buffer_size % 5 == 0 else 5 - buffer_size % 5),
               'constant')
    img = np.zeros((256, 256, 3), dtype=np.int)
    for i in range(0, len(b) - 1, 5):
        img[b[i], b[i + 1], 0] = (img[b[i], b[i + 1], 0] + b[i + 2]) % 255
        img[b[i], b[i + 1], 1] = (img[b[i], b[i + 1], 1] + b[i + 3]) % 255
        img[b[i], b[i + 1], 2] = (img[b[i], b[i + 1], 2] + b[i + 4]) % 255
    return Image.fromarray(img.astype('uint8')).convert('RGB')


def getMarkovMatrix(buffer, buffer_size):
    img = np.zeros((256, 256), dtype=np.int)
    for i in range(buffer_size - 2):
        img[buffer[i], buffer[i + 1]] += 1
    return Image.fromarray(img.astype('uint8'))


def file2Matrix(buffer, buffer_size):  #将文件转换为概率矩阵的方法

    bm = getBytesMatrix(buffer, buffer_size)
    #Image.fromarray(np.uint8(bm)).show()

    lm = getLocMatrix(buffer, buffer_size)
    #Image.fromarray(np.uint8(lm)).show()

    mm = getMarkovMatrix(buffer, buffer_size)
    #Image.fromarray(np.uint8(mm)).show()
    return bm, lm, mm
