## Отбор признаков для линейной регрессии

В ноутбуке Linear Regression.ipynb в папке семинаров в конце есть датасет с характеристиками домов и их ценами. Работаем только с отобранными действительными признаками (`num_features`).

1. Нужно реализовать линейную регрессию итеративным методом в матричной форме.
2. C помощью коэфициента детерминации, $F$ статистики и уровня значимости отобрать 5 наиболее полезных признаков (некоторые признаки стоит предобработать).
3. Оценить свое решение на тестовой выборке по метрике RMSE.

Все решение должно быть в отдельном ноутбуке состоящим из 4 частей:

1) реализация линейной регрессии и демонстрация работы на искусственом примере
2) предобработка и разбиение датасета
3) отбор признаков
4) оценка модели на отложенной выборке.