import numpy as np
from PIL import Image
from Work_With_Parameters import get_parameters
from Train_Neural_Network import predict


parameters = get_parameters()


def check_image(image, parameters=parameters):
    """
    Возвращает предсказанное значения для полученной фотографии
    Args:
        image: фотография, на котором определяется наличие человека
        parameters: параметры нейросети используемые для распознавания
    Returns:
        prediction: предсказанное значение
    """
    size_x = 160
    size_y = 90
    size = (size_x, size_y)
    image = Image.open(image)
    image = image.resize(size)
    array = np.array(image, dtype='uint8')
    array = array.reshape((1, size_x * size_y * 3)).T / 255
    data_test = array
    prediction = predict(data_test, parameters)
    return prediction
