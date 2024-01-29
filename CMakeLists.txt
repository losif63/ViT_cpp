cmake_minimum_required(VERSION 3.15)
# set project name and version
project(vit VERSION 1.0)

set(linux_system "$<PLATFORM_ID:Linux,Ubuntu>")
set(mac_system "$<PLATFORM_ID:Darwin,MacOS>")

# Change this to the location where torch or libtorch is installed

set(CMAKE_PREFIX_PATH $ENV{TORCH_PATH})
message(${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE cxx_std_17)

configure_file(Config.h.in Config.h)

add_subdirectory(lib)

add_executable(vit main.cpp)

target_compile_options(vit PRIVATE -pthread -lpthread)
target_link_libraries(vit PUBLIC "${TORCH_LIBRARIES}" compiler_flags vit_pytorch_cpp loaders tqdm PRIVATE pthread)

# set_property(TARGET vit PROPERTY CXX_STANDARD 17)