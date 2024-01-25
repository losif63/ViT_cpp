cmake_minimum_required(VERSION 3.15)

# set project name and version
project(vit VERSION 1.0)

find_package(Torch REQUIRED)

add_executable(vit vit.cpp)

target_compile_options(vit PRIVATE -pthread -lpthread)
target_link_libraries(vit PUBLIC "${TORCH_LIBRARIES}" PRIVATE pthread)

set_property(TARGET vit PROPERTY CXX_STANDARD 17)