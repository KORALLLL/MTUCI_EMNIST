import json #test

# Загрузка меток из JSON-файла
labels = {}
with open('dataset/stage3/labels10.csv', mode='r') as json_file:
    labels = json.load(json_file)
# Индекс текущего изображения
current_image_index = 0

iterator = 0

# Расположение файла
file_name = 'dataset/stage4/final_labels10.csv'

import cv2
import numpy as np

new_labels = {}
scale_factor = 20

# Путь к каталогу с изображениями
image_dir = 'dataset/stage2'

# Получение ключей (названий изображений) из словаря меток
image_filenames = list(labels.keys())
image_labels = list(labels.values())



# Создаем окно OpenCV
cv2.namedWindow("Image Window")

# Создание изображения с областью ввода
input_area = np.full((100, 400, 3), 255, dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
input_text = ""

while iterator < len(image_filenames):
    image_filename = image_filenames[iterator]
    
    # Загрузка изображения с помощью OpenCV
    image_path = f'{image_dir}/{image_filename}'
    image = cv2.imread(image_path)
    enlarged_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    
    # Отображение изображения в окне с меткой
    cv2.imshow("Image Window", enlarged_image)

    # Отображение области ввода с текущим текстом
    input_area = np.full((700, 560, 3), 255, dtype=np.uint8)
    cv2.putText(input_area, image_labels[iterator], (1, 500), font, 20, (0, 0, 0), 30, cv2.LINE_AA)
    cv2.imshow('Input Area', input_area)

    # Установка позиции окна с изображением
    cv2.moveWindow('Image Window', 0, 0)

    # Установка позиции окна с областью ввода
    cv2.moveWindow('Input Area', enlarged_image.shape[1], 0)
    
    # Ожидание нажатия клавиши
    key = cv2.waitKey(0)
    
    # Если нажата клавиша "q", выход из цикла
    if key == ord('q'):
        break
    
    # Переход к следующему изображению по нажатию клавиши "Enter"
    elif key == 13:  # Код клавиши "Enter"
        new_labels[image_filenames[iterator]] = image_labels[iterator]
        current_image_index += 1
        iterator += 1

    elif key ==8:
        iterator +=1

    print('index', current_image_index)

# Закрытие окна OpenCV
cv2.destroyWindow("Image Window")


# Запись в файл
with open(file_name, "w") as json_file:
    json.dump(new_labels, json_file)

