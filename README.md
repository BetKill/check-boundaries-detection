# Check Boundaries Detection

Проект для детектирования границ чеков и получения их координат с использованием OpenCV в C++.

##  Описание
Данный проект позволяет загружать изображения чеков, находить их границы и получать координаты с использованием компьютерного зрения. В основе алгоритма лежат методы обработки изображений, такие как пороговая обработка, контурный анализ и аппроксимация полигонов.

##  Установка и сборка
### 1. Установите зависимости
Перед сборкой убедитесь, что у вас установлены:
- **CMake** (минимальная версия 3.10)
- **OpenCV** (минимальная версия 4.5)
- **Компилятор C++** (GCC, Clang или MSVC)

### 2. Склонируйте репозиторий
```sh
git clone https://github.com/BetKill/check-boundaries-detection.git
cd check-boundaries-detection
```

### 3. Соберите проект
```sh
mkdir build
cd build
cmake ..
make
```

##  Использование
После сборки исполняемый файл будет находиться в папке `build/`. Запустите его с изображением на вход:
```sh
./check_boundaries_detector path/to/image.jpg
```
Программа обработает изображение, отобразит найденные границы и сохранит результат в `output/`.

##  Структура проекта
```
check-boundaries-detection/
│── images/                 # Примеры изображений чеков
│── output/                 # Сохранённые результаты работы
│── annotation.json         # JSON с разметкой данных
│── main.cpp                # Основной код программы
│── CMakeLists.txt          # CMake-скрипт сборки
│── Doxyfile                # Конфигурация для Doxygen
│── .gitignore              # Игнорируемые файлы
│── README.md               # Этот файл
```

##  Документация
Документация проекта сгенерирована с помощью Doxygen. Чтобы пересоздать её, выполните:
```sh
doxygen Doxyfile
```
После этого откройте `output/index.html` в браузере.

## Контакты
Если у вас есть вопросы или предложения, свяжитесь со мной через Issues в репозитории или по email: `betkill@mail.ru`.

