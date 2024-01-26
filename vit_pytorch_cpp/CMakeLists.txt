add_library(vit_pytorch_cpp vit.cpp)

target_include_directories(vit_pytorch_cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(vit_pytorch_cpp PUBLIC "${TORCH_LIBRARIES}" compiler_flags)