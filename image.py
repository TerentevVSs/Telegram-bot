import cv2
import numpy as np
def determinant(namelist, height, width, j, n):
    # функция выводит вес(на пиксель), и координаты центра масс изменения между двумя изображениями в списке под номерами j, n
    M = 0
    y = height / 2
    x = width / 2
    for i in range(height):
        for k in range(width):
            rj, gj, bj = namelist[j][i][k]
            rn, gn, bn = namelist[n][i][k]
            m = ((rj - rn) ** 2 + (gj - gn) ** 2 + (bj - bn) ** 2) ** 0.5
            M = M + m
            x = x + k * m
            y = y + i * m
    y = y / M
    x = x / M
    M = M/(height*width)
    return [M, x, y]
def rgb(namephoto, namelist, i):
    # функция записывает на i место в список изображение ввиде значения rgb
    img = cv2.imread(namephoto)
    namelist.insert(i, np.asarray(img))


