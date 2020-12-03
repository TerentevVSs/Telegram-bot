import cv2
import numpy as np


def determinant(namelist, height, width, j, n):
    """
    Функция выводит геометрическую длину изменения(на пиксель),
    и координаты центра изменения между двумя изображениями
    (под номерами j, n) в списке(namelist).
    Args:
        namelist: список с изображениями в попиксельном [r, g, b] виде
        height: высота изображения
        width: длина изображения
        j: номер 1-ого изображение в списке
        n: номер 2-ого изображение в списке

    Returns: [Len, x, y]
        Len:геометрическая длина изменения(на пиксель)
        x:координата центра изменения по горизонтали
        y:координата центра изменения по вертикали
    """
    Len = 0
    y = 0
    x = 0
    for i in range(height):
        for k in range(width):
            # rj, gj, bj - значения пикселя с координатами
            # i,k изображения под номером j
            # delta - длина изменения (i,k) пикселя
            rj, gj, bj = namelist[j][i][k]
            rn, gn, bn = namelist[n][i][k]
            delta = ((rj - rn) ** 2 + (gj - gn) ** 2 +
                     (bj - bn) ** 2) ** 0.5
            Len = Len + delta
            x = x + k * delta
            y = y + i * delta
    y = y / Len
    x = x / Len
    Len = Len / (height * width)
    return [Len, x, y]


def rgb(name_photo, namelist, i):
    """
    Функция записывает на i место в список(namelist) изображение
    (name_photo) в попиксельном виде значений [r,g,b]
    Args:
        name_photo: название изображения
        namelist: список в который будет занесено изображение в
        попиксельном виде
        i: номер места на который изображение будет занесено в список

    Returns:

    """
    ''' функция записывает на i место в список(namelist) изображение
    (name_photo) в попиксельном виде значений [r,g,b]
    Args: name_photo, namelist, i)
    name_photo:названия изображения
    namelist:список в который будет занесено изображение в 
    попиксельном виде
    i:номер места на который изображение будет занесено в список'''
    img = cv2.imread(name_photo)
    namelist.insert(i, np.asarray(img))
