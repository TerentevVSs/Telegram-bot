import cv2
import numpy as np
def determinant(namelist, height, width, j, n):
    ''' функция выводит геометрическую длину изменения(на пиксель),
    и координаты центра изменения между двумя изображениями(под номерами j, n) в списке(namelist).
    height и width разрешение изображений в списке'''
    Len = 0
    y = 0
    x = 0
    for i in range(height):
        for k in range(width):
            #rj, gj, bj - значения пикселя с координатами i,k изображения под номером j
            rj, gj, bj = namelist[j][i][k]
            rn, gn, bn = namelist[n][i][k]
            delta = ((rj - rn) ** 2 + (gj - gn) ** 2 + (bj - bn) ** 2) ** 0.5
            Len = Len + delta
            x = x + k * delta
            y = y + i * delta
    y = y / Len
    x = x / Len
    Len = Len/(height*width)
    return [Len, x, y]
def rgb(namephoto, namelist, i):
    ''' функция записывает на i место в список(namelist) изображение(namephoto) в попиксельном виде значений [r,g,b]'''
    img = cv2.imread(namephoto)
    namelist.insert(i, np.asarray(img))
