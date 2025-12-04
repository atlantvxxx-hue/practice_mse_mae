import numpy as np

x = np.array([1, 2, 3]) #кол-во кг
y = np.array([100, 190, 280]) # реальная цена

# начальные предсказания
w = 70
b = 30

#скорость обучения

learning_rate = 0.1

print("НАЧАЛО ОБУЧЕНИЯ: Угадываем цену яблок")

for iteration in range(5):
    print('='*30)
    print('итераций номер {},  показатели w = {}, b = {}'.format(iteration, w, b))

    y_pred = w*x + b
    print('предсказание {}'.format(y_pred))

    errors = y_pred-y
    print('ошибки {}'.format(errors))

    mae = np.mean(errors)
    print('Mse = {}'.format(mae))

    n = len(x)

    graw_w = 2/n * np.sum(errors * x)
    graw_b = 2 / n * np.sum(errors)

    w = w - learning_rate * graw_w
    b = b - learning_rate * graw_b

    final_predict = w * x + b
    print(final_predict)

for i in range(len(x)):
    error = final_predict[i] - y[i]
    print('')
    print(f'{x[i]:<10}, {y[i]:<10}, {error:<10}')
print('{:.2f}  и  {:.2f}'.format(w, b))