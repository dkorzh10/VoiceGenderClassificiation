# VoiceGenderClassificiation
Some simple attempts to voice gender classification with PyTorch on CommonVoice Ru dataset

[Link](https://drive.google.com/file/d/1szGLy15Dnuqj23P0CmsQWjE1AzJJMAYB/view?usp=sharing) to the best model'scheckpoint.

## Подготовка к запуску
Создаем окружение:
`conda create -n voiceclf python=3.8.12`


не пытаемся сразу делать `conda activate voiceclf`, закрываем всё, заходим заново и делаем `conda activate voiceclf` :)

Склонируйте репозиторий, перейдите в образованную директорию с данными репозитория, из которой будете работать, скачайте и положите туда датасет common voice corpus 7.0 Ru [link](https://commonvoice.mozilla.org/ru/datasets).


Разархиврируйте его через команду `tar -xvf cv-corpus-7.0-2021-07-21-ru.tar.gz`.


Установите необходимые библиотеки через `pip install -r requirements.txt`.

## Описание файлов

В целом, необходимые для обучения, валидации и тестирования датасеты готовы и есть в репозитории (`*.csv` файлы). Подробности их создания есть в 'EDP_and_nonworking_baselines.ipynb'. Однако, подготовку необходимых датафреймов для датасетов можно кастомизировать в 'datasets.py'. Можно выполнить `python datasets.py`.

Соотвественно, для обучения можно модели необходимо выполнить `python train.py`.

Разобраться, что происходит внутри момогут файлы `Final.ipynb` и `SmthWithSpectrograms.ipynb`.

## Discussion

Voice Activity detection, real-time inference не были сделаны. Никаких аугментацйи тоже добавлено не было, для балансировки train и val датаcетов просто брались примерно поровну женские и мужские записи (2556 женских, 3000 мужских). Можно было, например, "нарезать"  и западить до нужной длины записи с женскими голосами для балансировки. Аугментации данных (шумами и прочим) не было.

**Результаты:** accuracy 96.8%, ROC-AUC score 0.97 (не делалась специально для фиксированного шага). Есть проблема на тесте с классификаций женских фраз: false positive

![confusion_matrix](https://user-images.githubusercontent.com/31968272/148693498-de755a96-27bc-4e7a-9e5a-8f459bebc7ee.png)
