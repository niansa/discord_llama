cmake_minimum_required(VERSION 3.5)

project(discord_llama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(llama.cpp)
add_subdirectory(DPP)

add_executable(discord_llama main.cpp)
target_link_libraries(discord_llama PUBLIC dpp pthread)

install(TARGETS discord_llama
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
