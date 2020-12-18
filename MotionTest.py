import cv2
import module_image as image
import Neural_Network
import numpy as np
count = 0
couple = [0] * 2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Параметры сжатия m,n
m = 20
n = 20
# параметр количества кадров в усреднении len_static_set
# и массив кадров для усреднения  static_set  + средний кадр
len_static = 40
static_set = [0] * (len_static + 1)
# позиция в static_set на который будет поставлен новый статичный кадр
position_static_set = 0
# Проверяем, открывается ли видео
if not cap.isOpened():
    print("Error opening video stream or file")
# neural_network_set кадр для проверки
neural_network_check = "None"
# длина списка набор для нейросети
len_neural_set = 0
# medium_delta - среднее отклонение в покое
medium_delta = 0
# величина для подсчёта medium_delta
sum_delta = 0
# среднее отклонение двух последовательных кадров друг от друга
medium_duo = 0
# кол-во кадров после конца движения
relax_time = 0
# статус движение - покой
it_moves = False
# Статус движения после обработки нейросетью
worry = 0
# Словарь параметров дня и ночи
day = {"delta_high": 70, "duo_high": 5, "delta_low": 50, "duo_low": 2}
night = {"delta_high": 40, "duo_high": 0, "delta_low": 15, "duo_low": 2}
# Булева функция дня и ночи
it_is_day = True
# Открываем видео для чтения до окончания процесса
while cap.isOpened():
    ret, frame = cap.read()
    if it_is_day:
        params = day
    else:
        params = night
    if ret:
        count += 1
        if count > len_static + 1:
            old_frame = couple[1]
            couple[1] = image.rgb(frame, m, n)
            Delta = image.delta(couple)
            delta_duo = image.delta([couple[1], old_frame])
            if Delta > medium_delta*(params["delta_high"] +
                                     medium_duo/delta_duo) \
                    and delta_duo > medium_duo * params["duo_high"]:
                it_moves = True
                if count % 3 == 0:
                    neural_network_check = frame
                    neural_network_check = np.array(
                        neural_network_check, dtype='uint8')
                    neural_network_check = cv2.resize(
                        neural_network_check, dsize=(160, 90),
                        interpolation=cv2.INTER_CUBIC)
                    neural_network_check = neural_network_check.reshape(
                        160*90*3, 1) / 255
                    worry = Neural_Network.check_image(neural_network_check)
                    if worry == 1:
                        # Тут вывод в бота
                        print("Человек", count)
            else:
                # проверка было ли движение
                if it_moves:
                    # движение было, проверяет завершённость
                    relax_time += 1
                    if relax_time == 10:
                        it_moves = False
                else:
                    # движение не было иди прошло, проверка для
                    # добавление кадр в усреднение
                    if Delta < params["delta_low"] * medium_delta\
                            and delta_duo < medium_duo * params["duo_low"]:
                        # кадр подходит для усреднения
                        static_set[len_static] = static_set[len_static] \
                                     * len_static \
                                    - static_set[position_static_set][0]
                        static_set[position_static_set] = [couple[1],
                                                           Delta]
                        static_set[len_static] = (static_set[len_static] +
                                    static_set[position_static_set][0]) /\
                                                 len_static
                        couple[0] = static_set[len_static]
                        position_static_set = (position_static_set + 1) \
                                              % len_static
        elif count == len_static + 1:
            static_set[count - 2] = [couple[1], Delta]
            # вычислениек medium_delta
            sum_delta = sum_delta + Delta
            medium_delta = sum_delta / len_static
            # обновление среднего кадра
            static_set[len_static] = static_set[len_static] * (count - 2)
            static_set[len_static] += static_set[count - 2][0]
            static_set[len_static] = static_set[len_static] / (count - 1)
            couple[0] = static_set[len_static]
            couple[1] = image.rgb(frame, m, n)
            Delta = image.delta(couple)
            # вычисление medium_duo
            medium_duo += image.delta([static_set[count - 2][0],
                                       static_set[count - 3][0]])
            medium_duo = medium_duo / len_static
        elif count == 1:
            height, width = frame.shape[:2]
            couple[0] = image.rgb(frame, m, n)
        else:
            couple[1] = image.rgb(frame, m, n)
            Delta = image.delta(couple)
            static_set[count - 2] = [couple[1], Delta]
            # подсчёт
            if count > 2:
                medium_duo += image.delta([static_set[count - 2][0],
                                           static_set[count-3][0]])
            sum_delta = sum_delta + Delta
            # обновление среднего кадра
            static_set[len_static] = static_set[len_static] * (count - 2)
            static_set[len_static] += static_set[count - 2][0]
            static_set[len_static] = static_set[len_static] / (count - 1)
            couple[0] = static_set[len_static]
        cv2.imshow('Frame', frame)
        # Press "space" on keyboard to  exit
        if cv2.waitKey(32) & 0xFF == ord(' '):
            break
    else:
        break

# Выход из видео, когда процесс окончен
cap.release()
cv2.destroyAllWindows()
