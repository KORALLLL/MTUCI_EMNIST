import cv2
import numpy as np


index = 2568 ##################################################################470
image = cv2.imread('dataset/images/Scan2.jpg')

# Глобальные переменные
drawing = False
points = []
element_index = 0
zoom = 1.0
pan_x, pan_y = 0, 0


# Функция для преобразования точек под учетом масштаба и панорамирования
def transform_points(points, zoom, pan_x, pan_y):
    return [(int((x - pan_x) / zoom), int((y - pan_y) / zoom)) for x, y in points]

# Функция для обработки событий мыши
def mouse_callback(event, x, y, flags, param):
    global drawing, points, element_index, zoom, pan_x, pan_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(points) > 1:
            # Преобразуем точки под учетом масштаба и панорамирования
            transformed_points = transform_points(points, zoom, pan_x, pan_y)
            element_mask = np.full_like(original_image, (0,0,0)) # Важная маска по вырезанному символу
            
            cv2.drawContours(element_mask, [np.array(transformed_points)], -1, (255, 255, 255), thickness=cv2.FILLED)
            element_image = cv2.bitwise_and(original_image, element_mask)
            
            element_image[element_mask == 0] = 255

            # Находим координаты ограничивающего прямоугольника
            x, y, w, h = cv2.boundingRect(np.array(transformed_points))
            
            # Обрезаем изображение, чтобы оставить только непустые фрагменты
            cropped_element = element_image[y:y+h, x:x+w]
            
            cv2.imwrite(f'dataset/stage1/elements_{element_index + index}.png', cropped_element)
            element_index += 1
        points = []

# Создаем окно и устанавливаем обработчик событий мыши
cv2.namedWindow('Interactive Image Editor')
cv2.setMouseCallback('Interactive Image Editor', mouse_callback)

# Считываем изображение

original_image = image.copy()

while True:
    img_copy = image.copy()

    # Отображаем текущее изображение
    if zoom != 1.0:
        img_copy = cv2.resize(img_copy, None, fx=zoom, fy=zoom,  interpolation=cv2.INTER_NEAREST)
    if pan_x != 0 or pan_y != 0:
        img_copy = cv2.warpAffine(img_copy, np.float32([[1, 0, pan_x], [0, 1, pan_y]]), (img_copy.shape[1], img_copy.shape[0]))

    # Отрисовываем текущий контур
    if points:
        cv2.polylines(img_copy, [np.array(points)], isClosed=False, color=(0, 0, 255), thickness=2)

    cv2.imshow('Interactive Image Editor', img_copy)
    key = cv2.waitKey(1) & 0xFF

    # Обработка клавиш клавиатуры
    if key == ord('q'):
        break
    elif key == ord('z'):
        # Уменьшаем масштаб
        zoom -= 0.25
    elif key == ord('x'):
        # Увеличиваем масштаб
        zoom += 0.25
    elif key == ord('s'):
        # Перемещаем изображение вверх
        pan_y -= 100
    elif key == ord('d'):
        # Перемещаем изображение влево
        pan_x -= 100
    elif key == ord('w'):
        # Перемещаем изображение вниз
        pan_y += 100
    elif key == ord('a'):
        # Перемещаем изображение вправо
        pan_x += 100

# Закрываем окно OpenCV
cv2.destroyAllWindows()
