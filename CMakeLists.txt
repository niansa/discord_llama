cmake_minimum_required(VERSION 3.5)

project(discord_llama LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(anyproc)
add_subdirectory(DPP)
add_subdirectory(thread-pool)

add_executable(discord_llama main.cpp)
target_link_libraries(discord_llama PUBLIC dpp pthread libjustlm anyproc ggml threadpool)

install(TARGETS discord_llama
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
