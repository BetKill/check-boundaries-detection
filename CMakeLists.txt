cmake_minimum_required(VERSION 3.27)
project(kursv)

set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS ON)


# Автоматический поиск зависимостей через vcpkg
find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Создание исполняемого файла
add_executable(kursv main.cpp)

# Линковка библиотек
target_link_libraries(kursv PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)

# --- Добавление install целей ---
# Установка исполняемого файла в папку bin
install(TARGETS kursv
        RUNTIME DESTINATION bin/Debug
)
message(${CMAKE_INSTALL_PREFIX}, ${CMAKE_BINARY_DIR})
install(DIRECTORY "${CMAKE_BINARY_DIR}/Debug"
        DESTINATION bin
        FILES_MATCHING PATTERN "*.dll")


# Установка дополнительных файлов, если необходимо
install(FILES annotation.json DESTINATION share)
install(DIRECTORY images DESTINATION share)
