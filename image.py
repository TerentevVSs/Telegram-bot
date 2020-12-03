import numpy as np
def delta(namelist):
    '''Args: namelist
    Return: функция считает сумму квадратов разностей r,g,b по пикселям двух изображений'''
    C = np.add(namelist[0], -namelist[1])
    k = C.T
    С = np.dot(k, C)
    C = np.sum(C)
    return C
def rgb(name_photo, namelist, i, height, width):
    ''' Args: name_photo, namelist, i, height, width
    name_photo: название изображения
    namelist: список в который будет занесено изображение в
    попиксельном виде
    i: номер места на который изображение будет занесено в список
    height: размер изображение по вертикали
    width: размер изображение по горизонтали
    Returns: функция преобразует изображение(img) в вектор и записывает в массив(namelist) на i место'''
    array = np.array(name_photo, dtype='uint8').reshape((1, (height)*(width)*3)).T
    namelist.insert(i, array)
