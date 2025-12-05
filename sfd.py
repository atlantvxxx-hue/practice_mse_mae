import numpy as np

x = np.array([1, 2, 3]) #кол-во кг
y = np.array([100, 190, 280]) # реальная цена

# начальные предсказания
w = 70
b = 30

#скорость обучения

learning_rate = 0.1

print("НАЧАЛО ОБУЧЕНИЯ: Угадываем цену яблок")

for iteration in range(1, 6):
    print('='*30)
    print(f'итерация номер {iteration},  показатели w = {w:.1f}, b = {b:.1f}')

    y_pred = w*x + b
    print('предсказание : ', np.round(y_pred, 2))

    errors = y_pred-y
    print('ошибки : ', np.round(errors, 1))

    mae = np.mean(errors)
    print('Mse = ', np.round(mae, 2))

    #вычисляем градиенты
    n = len(x)

    graw_w = 2/n * np.sum(errors * x)
    graw_b = 2 / n * np.sum(errors)

    #делаем шаг оптимизации
    w = w - learning_rate * graw_w
    b = b - learning_rate * graw_b

    final_predict = w * x + b
    print('Финальное предсказание: ', np.round(final_predict, 2))

# конечный результат
for i in range(len(x)):
    error = final_predict[i] - y[i]
    print('кг=====цена=====ошибка')
    print(f'{x[i]:<5}, {y[i]:<5}, {error:<10.2f}')
print('аргументы w = {:.2f}  и  b = {:.2f}'.format(w, b))