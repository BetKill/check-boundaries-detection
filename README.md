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
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=[путь_к_vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

##  Использование
Выгрузить проект
```sh
cmake --install build --prefix "PATH_TO_INCLUDE" --config Debug 
```
Программа обработает изображение, отобразит найденные границы и сохранит результат в `output/`.
Открываем через терминал директорию с файлом(example.exe)
Пути указываем либо относительные, либо абсолютные, без ("").
'''
.\example.exe path_to_json path_to_directory example.txt
'''
##  Документация
Документация проекта сгенерирована с помощью Doxygen. Чтобы пересоздать её, выполните:
```sh
doxygen Doxyfile
```
После этого откройте `output/index.html` в браузере.

## Контакты
Если у вас есть вопросы или предложения, свяжитесь со мной через Issues в репозитории или по email: `betkill@mail.ru`.

