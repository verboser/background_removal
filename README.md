# Удаление фона

Простой веб-сервис для удаления фона с изображений. Закидываешь JPG, PNG или WEBP - получаешь PNG с прозрачным фоном. На фронте Streamlit показывает исходник и результат рядом.

Стек: FastAPI + Streamlit, под капотом rembg с моделью `birefnet-general` или прямой BiRefNet через PyTorch. Изображения обрабатывает Pillow.

## Запуск

```bash
conda env create -f environment.yml
conda activate background-removal
cp .env.example .env
uvicorn backend.main:app --reload --port 8000
```

Во втором терминале поднимаем фронт:

```bash
conda activate background-removal
streamlit run frontend/app.py
```

Проверить, что бэкенд живой:

```bash
curl http://127.0.0.1:8000/health
```

Если окружение уже создано и нужно только обновить зависимости:

```bash
conda env update -f environment.yml --prune
```

В `environment.yml` прописан Python 3.13. Локально гонял на `.venv` - всё работает 

## Настройка модели

Режим работы задается через `.env`:

```text
MODEL_BACKEND=rembg
REMBG_MODEL=birefnet-general
```

Для прямого BiRefNet:

```text
MODEL_BACKEND=birefnet
BIREFNET_MODEL_ID=ZhengPeng7/BiRefNet
BIREFNET_IMAGE_SIZE=512
DEVICE=cpu
```

После смены `MODEL_BACKEND` FastAPI надо перезапустить. Какой режим сейчас активен - видно в метаданных результата в Streamlit, например `Backend: rembg | Model: birefnet-general`

## Метрики

Если есть свой датасет с картинками и масками, можно прогнать MAE, IoU и Dice. В примере ниже `data/images` и `data/masks` это условные пути, их нужно заменить на реальные папки:

```bash
python ml/evaluate.py --images data/images --masks data/masks --backend rembg --out outputs/rembg.csv
python ml/evaluate.py --images data/images --masks data/masks --backend birefnet --out outputs/birefnet.csv
```

Скрипт пишет CSV по каждому файлу и короткий summary рядом с ним. Там есть средние метрики, худшие картинки, ошибка на границах, p50/p90 по времени и грубая оценка последовательной пропускной способности на CPU.

Для текущего Oxford Pets eval50:

```bash
python ml/evaluate.py --images data/oxford_pets/images_eval50 --masks data/oxford_pets/annotations/trimaps --file-list ml/oxford_pets_eval50.txt --backend rembg --out outputs/rembg_oxford_pets_eval50.csv --summary outputs/rembg_oxford_pets_eval50.summary.md --gallery outputs/rembg_oxford_pets_eval50_gallery.jpg --gallery-items 5 --strict
python ml/evaluate.py --images data/oxford_pets/images_eval50 --masks data/oxford_pets/annotations/trimaps --file-list ml/oxford_pets_eval50.txt --backend birefnet --out outputs/birefnet_oxford_pets_eval50.csv --summary outputs/birefnet_oxford_pets_eval50.summary.md --gallery outputs/birefnet_oxford_pets_eval50_gallery.jpg --gallery-items 5 --strict
```

`ml/oxford_pets_eval50.txt` фиксирует список файлов из test split. Галереи худших кейсов сохраняются в `outputs/` рядом с CSV и summary.

Если `annotations/trimaps` еще нет, я брал официальный архив так:

```bash
curl -L https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz -o data/oxford_pets/annotations.tar.gz
tar -xzf data/oxford_pets/annotations.tar.gz -C data/oxford_pets
```

Если `images_eval50` еще нет, он собирается из `images.tar.gz`:

```bash
python ml/prepare_eval50.py
```

В Oxford Pets trimap класс `2` это фон. Классы `1` и `3` в метриках считаются объектом.

Быстрые тесты:

```bash
pytest
```

Обычный прогон не грузит модели. Он проверяет расчет метрик и ошибки запуска evaluator: нет папки с картинками, пустая папка, нет пар картинка + маска, нет файла из `--file-list` при `--strict`.

Реальные модели можно проверить отдельно:

```bash
pytest -m integration
```

Этот прогон берет один файл из Oxford Pets и проверяет, что rembg и прямой BiRefNet дают маску с IoU выше 0.6 и Dice выше 0.75.

## Структура

```text
backend/     FastAPI, загрузка модели, обработка изображений
frontend/    Streamlit
ml/          расчет метрик и разбор ошибок по маскам
tests/       тесты метрик, ошибок evaluator и интеграционный прогон моделей
```

## Что стоит учесть

- CPU-инференс медленный. Если планируется больше одного-двух одновременных пользователей - нужен GPU или хотя бы очередь задач, иначе всё встанет.
- BiRefNet на CPU тоже не быстрый. Для сравнения моделей лучше ограничивать `--limit`, а полный прогон делать уже на GPU.
- Текущий прогон по Oxford Pets eval50 это не нагрузочный тест. По latency из summary получается примерно 0.19 изображения/сек для rembg и 0.60 изображения/сек для прямого BiRefNet на моей CPU-конфигурации.

Подробнее про выбор модели и компромиссы - в `report.md`.
