add_library(loaders dataloaders.cpp)

target_include_directories(loaders INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(loaders PUBLIC "${TORCH_LIBRARIES}" compiler_flags)