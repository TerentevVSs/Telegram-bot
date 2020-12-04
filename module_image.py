import numpy as np


def delta(namelist):
    '''Args: namelist
    Return: функция считает сумму квадратов разностей r,g,b по пикселям двух изображений'''
    C = np.add(namelist[0], -namelist[1])
    C[0][C[0] <= 0.7] = 0
    C = C.T
    C = np.dot(C, C.T)
    C = np.sum(C)
    return C


def rgb(name_photo, height, width):
    """
    функция преобразует изображение(img) в вектор
    Args:
        name_photo: название изображения
        height: размер изображение по вертикали
        width: размер изображение по горизонтали
    Returns:
        array: функция преобразует изображение(img) в вектор
    """
    array = np.array(name_photo, dtype='uint8').reshape(
        (1, (height) * (width) * 3)).T/255
    return array
