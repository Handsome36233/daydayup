cmake_minimum_required(VERSION 3.10)

# 项目名称
project(MyProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include_ort)
include_directories(${PROJECT_SOURCE_DIR}/include_ort_infer)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加库文件目录
link_directories(${PROJECT_SOURCE_DIR}/lib)

# 查找所有的 .cpp 文件（除了主函数文件）
file(GLOB SRC_FILES "src/*.cpp")

# 添加可执行文件
add_executable(demo main.cpp ${SRC_FILES})

# 链接依赖库和 OpenCV 库
target_link_libraries(demo ${OpenCV_LIBS})

# 链接依赖库和 libonnxruntime 库
target_link_libraries(demo ${PROJECT_SOURCE_DIR}/lib/libonnxruntime.so)
