cmake_minimum_required(VERSION 3.10)
project(ImageRetrieval)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tìm OpenCV
find_package(OpenCV REQUIRED)

# Include thư viện
include_directories(${OpenCV_INCLUDE_DIRS})

# Thêm thư mục include
include_directories(include)

# Gộp toàn bộ file .cpp
file(GLOB SOURCES src/*.cpp)

# Tạo file thực thi
add_executable(ImageRetrieval ${SOURCES})

# Liên kết thư viện
target_link_libraries(ImageRetrieval ${OpenCV_LIBS})
