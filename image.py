import cv2
import numpy as np
def determinant(namelist, height, width, j, n):
    ''' функция выводит геометрическую длину изменения(на пиксель),
    и координаты центра изменения между двумя изображениями(под номерами j, n) в списке(namelist).
    Args:namelist, height, width, j, n
    namelist:список с изображениями в попиксельном [r, g, b] виде
    height:высота изображения
    widht:длина изображения
    j:номер 1-ого изображение в списке
    n:номер 2-ого изображение в списке
    Return:[Len, x, y]
    Len:геометрическая длина изменения(на пиксель)
    x:координата центра изменения по горизонтали
    y:координата центра изменения по вертикали'''
    Len = 0
    y = 0
    x = 0
    for i in range(height):
        for k in range(width):
            #rj, gj, bj - значения пикселя с координатами i,k изображения под номером j
            #delta - длина изменения (i,k) пикселя
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
    ''' функция записывает на i место в список(namelist) изображение(namephoto) в попиксельном виде значений [r,g,b]
    Args:namephoto, namelist, i)
    namephoto:названия изображения
    namelist:список в который будет занесено изображение в попиксельном виде
    i:номер места на который изображение будет занесено в список'''
    img = cv2.imread(namephoto)
    namelist.insert(i, np.asarray(img))
