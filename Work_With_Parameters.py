import numpy as np


def save_parameter(parameter_name, parameters):
    """
    Сохраняет параметр нейросети в текстовый файл
    Args:
        parameter_name: сохраняемый параметр нейросети
        parameters: переменная из которой сохраняется параметр
    Returns:
        Создает файл "parameter parameter_name.txt" с параметром
    """
    with open("parameter %s.txt" % parameter_name, "w") as file:
        A = parameters[parameter_name]
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                file.write(str(A[i][j]) + ' ')
            file.write('\n')


def save_parameters(parameters):
    """
    Сохраняет все параметры нейросети каждый в свой текстовый файл
    """
    save_parameter("W1", parameters)
    save_parameter("W2", parameters)
    save_parameter("W3", parameters)
    save_parameter("b1", parameters)
    save_parameter("b2", parameters)
    save_parameter("b3", parameters)


def get_parameter(parameter_name):
    """
    Получает из текстового файла "parameter parameter_name.txt"
    параметер нейросети parameter_name
    Args:
        parameter_name: получаемый из текстового файла параметер
    Returns:
        parameter: полученный из текстового файла параметр нейросети
    """
    with open('parameter %s.txt' % parameter_name, "r") as file:
        count = 0
        for line in file:
            line = line.split(' ')
            line = line[:len(line) - 1]
            for i in range(len(line)):
                line[i] = float(line[i])
            line = np.array(line)
            if count == 0:
                len_line = len(line)
                data = np.array(line).reshape((1, len_line))
                count = 1
            else:
                data = np.concatenate(
                    (data, line.reshape((1, len_line))))
    parameter = data
    return parameter


def get_parameters():
    """
    Получает параметры нейросети каждый из своего текстового файла
    Returns:
        parameters: словарь из параметров нейросети
    """
    W1 = get_parameter("W1")
    W2 = get_parameter("W2")
    W3 = get_parameter("W3")
    b1 = get_parameter("b1")
    b2 = get_parameter("b2")
    b3 = get_parameter("b3")
    parameters = {"W1": W1, "W2": W2, "W3": W3, "b1": b1, "b2": b2,
                  "b3": b3}
    return parameters
