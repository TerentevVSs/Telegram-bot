import cv2
import module_image as image
import Neural_Network
import numpy as np
import telebot

from telebot import types

bot = telebot. TeleBot("1427672157:AAH_K0LbWsIuNyUR3vQC9tEvxxrkHlqVEuY")


# Булева функция дня и ночи
it_is_day = None
# Словарь параметров дня и ночи
day = {"delta_high": 40, "duo_high": 5, "delta_low": 50, "duo_low": 2}
night = {"delta_high": 20, "duo_high": 0, "delta_low": 15, "duo_low": 2}

keyboard1 = telebot.types.ReplyKeyboardMarkup(True, True)
keyboard1.row('Начнем', 'Не нужно')
keyboard2 = telebot.types.ReplyKeyboardMarkup(True, True)
keyboard2.row('День', 'Ночь')


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, начнем видеонаблюдение?', reply_markup=keyboard1)


@bot.message_handler(content_types=['text'])
def send_text(message):
    if message.text == 'Начнем':
        bot.send_sticker(message.chat.id, 'CAACAgIAAxkBAAKGY1_bde_LilNGDnjcXIBFL93uqlwEAAJiFgAC6VUFGKq9bs1J0e3SHgQ')
        bot.send_message(message.chat.id, 'Выберите режим', reply_markup=keyboard2)
        bot.register_next_step_handler(message, get_work)  # следующий шаг – функция  get_work
    elif message.text == 'Не нужно':
        bot.send_message(message.chat.id, 'Прощай')

def get_work(message):
    global it_is_day # Булева функция дня и ночи
    if message.text == 'День':
        it_is_day = True
    bot.send_sticker(message.chat.id, 'CAACAgIAAxkBAAKGY1_bde_LilNGDnjcXIBFL93uqlwEAAJiFgAC6VUFGKq9bs1J0e3SHgQ') # просто чтобы проверить работу кода
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
    # Проверяем, открывается ли видео
    if not cap.isOpened():
        bot.send_message(message.chat.id, 'Error opening video stream or file')
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
                if Delta > medium_delta * (params["delta_high"] +
                                            medium_duo / delta_duo) \
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
                            160 * 90 * 3, 1) / 255
                        worry = Neural_Network.check_image(neural_network_check)
                        if worry == 1:
                            # Тут вывод в бота
                            bot.send_message(message.chat.id, 'ЗАФИКСИРОВАН ЧЕЛОВЕК')
                            print('men') # проверка корректной работы программы
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
                        if Delta < params["delta_low"] * medium_delta \
                                and delta_duo < medium_duo * params["duo_low"]:
                            # кадр подходит для усреднения
                            static_set[len_static] = static_set[len_static] \
                                                        * len_static \
                                                        - static_set[position_static_set][0]
                            static_set[position_static_set] = [couple[1],
                                                                Delta]
                            static_set[len_static] = (static_set[len_static] +
                                                        static_set[position_static_set][0]) / \
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
                                                static_set[count - 3][0]])
                sum_delta = sum_delta + Delta
                # обновление среднего кадра
                static_set[len_static] = static_set[len_static] * (count - 2)
                static_set[len_static] += static_set[count - 2][0]
                static_set[len_static] = static_set[len_static] / (count - 1)
                couple[0] = static_set[len_static]
            #cv2.imshow('Frame', frame)
            # Press "space" on keyboard to  exit
            if message.text == '/stop':
                bot.send_message(message.chat.id, 'До свидания!')
                break
        else:
            break
        # Выход из видео, когда процесс окончен
    cap.release()
    cv2.destroyAllWindows()


@bot.message_handler(content_types = ['sticker'])
def sticker_id(message):
    print(message)

if __name__ == '__main__':
    bot.polling(none_stop=True)