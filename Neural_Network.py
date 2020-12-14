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
    prediction = predict(image, parameters)
    return prediction
