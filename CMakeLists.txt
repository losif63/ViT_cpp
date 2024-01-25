cmake_minimum_required(VERSION 3.15)
# set project name and version
project(vit VERSION 1.0)

# Change this to the location where torch or libtorch is installed
set(CMAKE_PREFIX_PATH /home/sslunder7/anaconda3/envs/fbgemm/lib/python3.8/site-packages/torch/)
find_package(Torch REQUIRED)

add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE cxx_std_17)

configure_file(Config.h.in Config.h)

add_subdirectory(Classes)

add_executable(vit vit.cpp)

target_compile_options(vit PRIVATE -pthread -lpthread)
target_link_libraries(vit PUBLIC "${TORCH_LIBRARIES}" compiler_flags Classes PRIVATE pthread)

# set_property(TARGET vit PROPERTY CXX_STANDARD 17)