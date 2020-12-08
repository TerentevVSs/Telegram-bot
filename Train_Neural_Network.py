import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import Work_With_Parameters as WWP


def initialize_parameters(layer_dims):
    """
    Инициализация параметров сети
    Args:
        layer_dims: список из количества нейронов в каждом слое
    Returns: словарь W и b
    Wl размера (layer_dims[l], layer_dims[l-1])
    bl размера (layer_dims[l], 1)
    """
    parameters = {}
    L = len(layer_dims)  # число слоев в сети
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],
                                                   layer_dims[
                                                       l - 1]) * 0.01 * np.sqrt(
            2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    """
    Прямое распространение(Вычисление Z в одном слое)
    Args:
        A: Значения в предыдущем слое размера
            (size of previous layer, number of examples)
        W: Матрица весов размера
            (size of current layer, size of previous layer)
        b: Матрица свободных коэффициентов размера
            (size of the current layer,1)
    Returns:
        Z = W * A + b
        cache: список из "A", "W" и "b" для обратного распространения
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def sigmoid(Z):
    """
    Функция активации сигмоид
    Args:
        Z: значение от которого вычисляется сигмоид
    Returns:
        A: значение сигмоида
        cache: значение Z для обратного распространения
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    """
    Функция активации ReLU
    Args:
        Z: значение от которого вычисляется сигмоид
    Returns:
        A: значение функции ReLU
        cache: значение Z для обратного распространения
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Прямое распространение (вычисление функции активации)
    Args:
        A_prev: Значения в предыдущем слое размера
            (size of previous layer, number of examples)
        W: Матрица весов размера
            (size of current layer, size of previous layer)
        b: Матрица свободных коэффициентов размера
            (size of the current layer,1)
        activation: тип функции активации (сигмоид или ReLU)
    Returns:
        A: резльтат вычисления функции активации
        cache: список из значений "A", "W", "b" и "Z" для обратого
        распространения
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    else:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)

    return A, cache


def L_layer_network_forward(X, parameters):
    """
    Прямое распространение в сети:
        l-1 слой с функций ReLU и 1 слой с сигмоидом, где l-число слоев
    Args:
        X: Входные данные (input size, number of examples)
        parameters: словарь из параметров W и b
    Returns:
        predict_value: предсказанное значение
        caches: значения из linear_activation_forward
    """
    caches = []
    A = X
    L = len(parameters) // 2  # число слоев в сети
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters['W' + str(l)],
                                             parameters['b' + str(l)],
                                             "relu")
        caches.append(cache)
    predict_value, cache = linear_activation_forward(A, parameters[
        'W' + str(L)],
                                                     parameters[
                                                         'b' + str(L)],
                                                     "sigmoid")
    caches.append(cache)
    return predict_value, caches


def compute_cost(predict_value, Y, parameters, lambd):
    """
    Вычисление функции потерь
    Args:
        predict_value: предсказанное значение
        Y: правильное значение
        parameters: словарь из параметров W и b
        lambd: параметр для L2 регуляризации
    Returns:
        cost: значение функции потерь
    """
    m = Y.shape[1]
    cost = -1 / m * np.sum(Y * np.log(predict_value + 10 ** (-10)) +
                           (1 - Y) * np.log(
        1 + 10 ** (-10) - predict_value))
    k = 1 / m * lambd / 2
    L = len(parameters) // 2  # число слоев в сети
    for l in range(1, L):
        cost = cost + k * np.sum(np.square(parameters['W' + str(l)]))
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache, lambd):
    """
    Обратное распространение
    Args:
        dZ: производная Z
        cache: значения из прямого распространения
        lambd: параметр для L2 регуляризации
    Returns:
        dA_prev: производная функции потерь по Z
        dW: производная функции потерь по W
        db: производная функции потерь по b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1 / m * np.dot(dZ, A_prev.T) + lambd / m * W
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def relu_backward(dA, cache):
    """
    Обратное распространение в слое с функцией активации ReLU
    Args:
        dA: градиент после активации
        cache: значение Z для обратного распространения
    Returns:
        d: производная функции потерь по Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, cache):
    """
    Обратное распространение в слое с функцией активации сигмоид
    Args:
        dA: градиент после активации
        cache: значение Z для обратного распространения
    Returns:
        d: производная функции потерь по Z
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def linear_activation_backward(dA, cache, activation, lambd):
    """
    Обратное распространение
    Args:
        dA: градиент после активации для текущего слоя
        cache: значение для обратного распространения
        activation: функция активации сигмоид или ReLU
        lambd: параметр для L2 регуляризации
    Returns:
        dA_prev: производная функции потерь по Z
        dW: производная функции потерь по W
        db: производная функции потерь по b
    """
    dA_prev = []
    dW = []
    db = []
    if activation == "relu":
        dZ = relu_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, cache[0], lambd)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, cache[0], lambd)
    return dA_prev, dW, db


def L_layer_network_backward(predict_value, Y, caches, lambd):
    """
    Обратное распространение в сети
    Args:
        predict_value: предсказанное значение
        Y: правильное значение
        caches: значения dW и db
        lambd: параметр для L2 регуляризации
    Returns:
        grads: словарь производных dA, dW, db
    """
    grads = {}
    L = len(caches)  # количество слоев в сети
    Y = Y.reshape(predict_value.shape)
    dpredict_value = - (np.divide(Y, predict_value + 10 ** (-10)) -
                        np.divide(1 - Y,
                                  1 + 10 ** (-10) - predict_value))
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads[
        "db" + str(L)] = linear_activation_backward(dpredict_value,
                                                    current_cache,
                                                    "sigmoid", lambd)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu", lambd)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Обновление параметров (градиентный спуск)
    Args:
        parameters: словарь параметров W и b
        grads: словарь содержащий значение производных dA, dW, db
        learning_rate: коэффициент шага спуска
    Returns:
        parameters: обновленный словарь значений dW, db
    """
    L = len(parameters) // 2  # количество слоев
    for l in range(L):
        parameters["W" + str(l + 1)] -= grads["dW" + str(
            l + 1)] * learning_rate
        parameters["b" + str(l + 1)] -= grads["db" + str(
            l + 1)] * learning_rate
    return parameters


def L_layer_network(X, Y, layers_dims, learning_rate=0.,
                    num_iterations=0.,
                    print_cost=False,
                    lambd=0):
    """
    Реализация L-слойной сети
    Args:
        X: входные значения
        Y: правильные значения
        layers_dims:
        learning_rate: коэффициент шага спуска
        num_iterations: количество повторений алгоритма
        print_cost: печать функции потерь
        lambd: параметр для L2 регуляризации
    Returns:
        parameters: итоговые значение W и b
    """
    costs = []
    parameters = initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        predict_value, caches = L_layer_network_forward(X, parameters)
        cost = compute_cost(predict_value, Y, parameters, lambd)
        grads = L_layer_network_backward(predict_value, Y, caches,
                                         lambd)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i:" % i, cost)
        if print_cost and i % 100 == 0:
            costs.append(cost)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


def predict_training(X, Y, parameters):
    """
    Предсказание результата для данного X и проверка точнсти предсказаний
    Args:
        X: Входные значения
        Y: Правильные значения
        parameters: значение W и b
    Returns:
        p: Точность модели
    """
    m = X.shape[1]
    p = np.zeros((1, m))
    probabilities, caches = L_layer_network_forward(X, parameters)
    print('prob=', probabilities)
    print('Y=', Y)
    for i in range(0, probabilities.shape[1]):
        if probabilities[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("Accuracy: " + str(np.sum((p == Y) / m)))
    return p


def predict(X, parameters):
    """
    Предсказание результата для данного X
    Args:
        X: Входные значения
        parameters: значение W и b
    Returns:
        Возвращает предсказанное значение
    """
    probabilities, caches = L_layer_network_forward(X, parameters)
    if probabilities[0][0] > 0.5:
        return 1
    else:
        return 0


if __name__ == "__main__":
    print("Это режим разработчика.")
    print("""Запуск этого файла приведет к перезаписи параметров 
нейросети. Если у Вас нет в этой папке тренировчных файлов,
отменить запуск программы нажав "N", иначе параметры нейросети
обнулятся, что приведет к потере работоспособности. Если 
хотите продолжить нажмите "Y", если нет нажмите "N".""")
    check = input("Y/N?:")
    """
    Данный код предназначен для быстрой настройки сети и будет удален 
    позднее.
    """
    if check == "Y" or check == "y":
        size_x = 160
        size_y = 90
        img = Image.open('test (1).jpg')
        size = (size_x, size_y)
        img = img.resize(size)
        array = np.array(img, dtype='uint8').reshape(
            (1, size_x * size_y * 3)).T
        data_train = np.array(array)
        label_train = [1]

        for i in range(2, 26):
            img = Image.open('test (%i).jpg' % i)
            img = img.resize(size)
            array = np.array(img, dtype='uint8')
            array = array.reshape((1, size_x * size_y * 3)).T / 255
            data_train = np.concatenate((data_train, array), axis=1)
            label_train.append(1)

        for i in range(1, 26):
            img = Image.open('nontest (%i).jpg' % i)
            img = img.resize(size)
            array = np.array(img, dtype='uint8')
            array = array.reshape((1, size_x * size_y * 3)).T / 255
            data_train = np.concatenate((data_train, array), axis=1)
            label_train.append(0)

        for i in range(1, 27):
            img = Image.open('n (%i).jpg' % i)
            img = img.resize(size)
            array = np.array(img, dtype='uint8')
            array = array.reshape((1, size_x * size_y * 3)).T / 255
            data_train = np.concatenate((data_train, array), axis=1)
            label_train.append(0)

        for i in range(1, 27):
            img = Image.open('y (%i).jpg' % i)
            img = img.resize(size)
            array = np.array(img, dtype='uint8')
            array = array.reshape((1, size_x * size_y * 3)).T / 255
            data_train = np.concatenate((data_train, array), axis=1)
            label_train.append(1)

        img1 = Image.open('a (1).jpg')
        img1 = img1.resize(size)
        array1 = np.array(img1, dtype='uint8')
        array1 = array1.reshape((1, size_x * size_y * 3)).T / 255
        data_test = array1
        label_test = [1]

        img2 = Image.open('a (2).jpg')
        img2 = img2.resize(size)
        array2 = np.array(img2, dtype='uint8')
        array2 = array2.reshape((1, size_x * size_y * 3)).T / 255
        data_test = np.concatenate((data_test, array2), axis=1)
        label_test.append(1)

        img3 = Image.open('b (1).jpg')
        img3 = img3.resize(size)
        array3 = np.array(img1, dtype='uint8')
        array3 = array3.reshape((1, size_x * size_y * 3)).T / 255
        data_test = np.concatenate((data_test, array3), axis=1)
        label_test.append(1)

        img4 = Image.open('b (2).jpg')
        img4 = img4.resize(size)
        array2 = np.array(img2, dtype='uint8')
        array2 = array2.reshape((1, size_x * size_y * 3)).T / 255
        data_test = np.concatenate((data_test, array2), axis=1)
        label_test.append(1)

        img3 = Image.open('c (1).jpg')
        img3 = img3.resize(size)
        array3 = np.array(img1, dtype='uint8')
        array3 = array3.reshape((1, size_x * size_y * 3)).T / 255
        data_test = np.concatenate((data_test, array3), axis=1)
        label_test.append(1)

        img4 = Image.open('c (2).jpg')
        img4 = img4.resize(size)
        array4 = np.array(img4, dtype='uint8')
        array4 = array4.reshape((1, size_x * size_y * 3)).T / 255
        data_test = np.concatenate((data_test, array4), axis=1)
        label_test.append(1)

        label_train = np.array(label_train).reshape((1, 102))
        label_test = np.array(label_test).reshape((1, 6))
        layers_dims = [size_x * size_y * 3, 6, 4, 1]
        parameters = L_layer_network(data_train, label_train,
                                     layers_dims,
                                     num_iterations=2000,
                                     learning_rate=0.01,
                                     print_cost=True, lambd=0)
        predict_train = predict_training(data_train, label_train,
                                         parameters)
        print(predict_train)
        predict_test = predict_training(data_test, label_test,
                                        parameters)
        print(predict_test)

        WWP.save_parameters(parameters)
        parameters = WWP.get_parameters()
    else:
        pass
