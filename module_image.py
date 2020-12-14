import numpy as np
import cv2
def delta(namelist):
    """
    Args: 
        namelist - список кадров
    Return: 
        сумму квадратов разностей r,g,b по пикселям двух изображений
    """
    S = np.add(namelist[0], -namelist[1])
    S[0][S[0] <= 0.7] = 0
    S = S.T
    S = np.dot(S, S.T)
    S = np.sum(S)
    return S


def rgb(name_photo, m, n):
    """
    функция преобразует изображение(img) в вектор
    Args:
        name_photo: название изображения
        height: размер изображение по вертикали
        width: размер изображение по горизонтали
    Returns:
        array: функция преобразует изображение(img) в вектор
    """
    size = (m, n)
    name_photo = cv2.resize(name_photo, dsize=size, interpolation=cv2.INTER_CUBIC)
    array = np.array(name_photo, dtype='uint8').reshape((1, m * n * 3)).T/255
    return array
