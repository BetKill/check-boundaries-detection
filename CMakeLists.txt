cmake_minimum_required(VERSION 3.27)
project(kursv)

set(CMAKE_CXX_STANDARD 17)

# Подключаем vcpkg
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
endif()

# Путь к установленному OpenCV через vcpkg
set(OpenCV_DIR "C:/Users/betki/.vcpkg-clion/vcpkg/installed/x64-mingw-dynamic/share/opencv4")

# Поиск библиотек
find_package(OpenCV REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Создание исполняемого файла
add_executable(kursv main.cpp)

# Линковка библиотек
target_link_libraries(kursv PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
