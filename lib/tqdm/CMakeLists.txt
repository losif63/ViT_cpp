add_library(tqdm tqdm.h)
set_target_properties(tqdm PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(tqdm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tqdm compiler_flags)