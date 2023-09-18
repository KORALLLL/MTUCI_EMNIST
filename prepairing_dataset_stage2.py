# Начальный индекс изображения
old_image_index = 1360 ##################################################################
new_image_index = 1814 ##################################################################


import cv2
import numpy as np

# Путь к папке с изображениями
image_folder = 'dataset/stage1/'

# Коэффициент увеличения пикселей
scale_factor = 20

while True:
    # Формируем имя файла
    image_filename = f'photo_element_{old_image_index}.png'
    image_path = image_folder + image_filename

    # Загрузка изображения
    image = cv2.imread(image_path)

    if image is not None:

        # Определение размера квадрата (берем максимальный размер между шириной и высотой)
        size = max(image.shape[0], image.shape[1])

        # Создание пустого квадратного изображения
        square_image = np.full((size, size, 3), 255, dtype=np.uint8)

        # Определение координат для вставки исходного изображения по центру квадрата
        x_offset = (size - image.shape[1]) // 2
        y_offset = (size - image.shape[0]) // 2

        # Вставка исходного изображения в центр квадрата
        square_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image

        # Конвертация в оттенки серого
        square_image = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)

        # Уменьшение яркости всех пикселей, кроме белых
        mask = square_image > 170
        square_image[mask] = 255  # Например, уменьшим яркость вдвое


        # Увеличение изображения
        enlarged_image = cv2.resize(square_image, (28, 28), interpolation=cv2.INTER_LINEAR)
        
        # Увеличение размера пикселей
        enlarged_image1 = cv2.resize(enlarged_image, (28 * scale_factor, 28 * scale_factor), interpolation=cv2.INTER_NEAREST)

        # Отображение увеличенного изображения
        cv2.imshow('Enlarged Image', enlarged_image1)

        # Ожидание нажатия клавиши
        key = cv2.waitKey(0)
        print('old index', old_image_index)

        # Если нажата клавиша 'Y', переходим к следующему изображению
        if key == ord('y'):
            old_image_index += 1

        # Если нажата клавиша 'U', сохраняем название изображения в массиве
        elif key == ord('u'):
            cv2.imwrite(f'dataset/stage2/new_image_{new_image_index}.png', enlarged_image)
            old_image_index += 1
            new_image_index += 1

        # Если нажата клавиша 'Esc', выходим из цикла
        elif key == ord('q'):
            break

    else:
        # Если изображение с заданным индексом не найдено, выходим из цикла
        break


# Закрываем окно OpenCV
cv2.destroyAllWindows()
