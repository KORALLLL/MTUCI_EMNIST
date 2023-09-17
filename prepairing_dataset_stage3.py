# Начальный индекс изображения
image_index = 3962
# Расположение файла
file_name = 'dataset/stage3/labels20.csv'


import cv2
import numpy as np
import json

# Путь к папке с изображениями
image_folder = 'dataset/stage2/'

# Создание пустого словаря для сохранения символов и их соответствующих изображений
symbol_image_dict = {}



# Коэффициент увеличения пикселей
scale_factor = 20

# Создание изображения с областью ввода
input_area = np.full((100, 400, 3), 255, dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
input_text = ""

while True:
    # Формируем имя файла
    image_filename = f'processed_{image_index}.png'
    image_path = image_folder + image_filename

    # Загрузка изображения
    image = cv2.imread(image_path)

    if image is not None:
        # Увеличение размера изображения
        enlarged_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

        # Отображение окна с изображением
        cv2.imshow('Image', enlarged_image)

        # Отображение области ввода с текущим текстом
        input_area = np.full((700, 560, 3), 255, dtype=np.uint8)
        cv2.putText(input_area, input_text, (1, 500), font, 20, (0, 0, 0), 30, cv2.LINE_AA)
        cv2.imshow('Input Area', input_area)

        # Установка позиции окна с изображением
        cv2.moveWindow('Image', 0, 0)

        # Установка позиции окна с областью ввода
        cv2.moveWindow('Input Area', enlarged_image.shape[1], 0)

        # Ожидание ввода символа
        key = cv2.waitKey(0)

        # Если введена буква или цифра, добавляем ее к введенному тексту
        if (key >= ord('a') and key <= ord('z')) or (key >= ord('0') and key <= ord('9')):
            input_text += chr(key)
        # Если нажата клавиша backspace, удаляем последний символ
        elif key == 8:  # 8 соответствует backspace
            input_text = input_text[:-1]
        # Если нажата клавиша enter, сохраняем символ и название изображения в словаре
        elif key == 13:  # 13 соответствует enter
            print(input_text, 'current index', image_index)
            symbol_image_dict[image_filename] = input_text
            input_text = ""
            image_index += 1  # Увеличиваем индекс изображения после сохранения символа

        # Если нажата клавиша 'Esc', выходим из цикла
        if key == 27:
            break

    else:
        # Если изображение с заданным индексом не найдено, выходим из цикла
        break



# Запись в файл
with open(file_name, "w") as json_file:
    json.dump(symbol_image_dict, json_file)



# Закрываем окно OpenCV
cv2.destroyAllWindows()
