import numpy as np

x = np.array([1, 3, 5, 7, 10]) # расстояние в км
y = np.array([15, 25, 35, 45, 60 ]) # время доставки в минутах

w = 20
b = 5

#скорость обучения
learning_rate = 0.01

for iteration in range(1,6):
    print('итерация номер {}, показатель w = {:.2f} и b = {:.2f}'.format(iteration, w, b))

    y_pred = w * x + b
    print('предсказание : ', np.round(y_pred, 1))

    errors = y_pred - y
    print('ошибки : ', np.round(errors, 1))

    mae = f'{abs(np.mean(errors)):.2f}'
    print('mae = {}'.format(mae))

    n = len(x)
    graw_w = 2/n * np.sum(errors * x)
    graw_b = 2/n * np.sum(errors)

    w = w - learning_rate * graw_w
    b = b - learning_rate * graw_b
