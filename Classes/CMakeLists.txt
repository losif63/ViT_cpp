add_library(Classes Classes.cpp)

target_include_directories(Classes INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(Classes PUBLIC "${TORCH_LIBRARIES}" compiler_flags)