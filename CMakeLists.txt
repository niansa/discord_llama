cmake_minimum_required(VERSION 3.5)

project(discord_llama LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ANYPROC_COSCHED ON CACHE BOOL "" FORCE)
set(ANYPROC_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LM_COSCHED ON CACHE BOOL "" FORCE)
set(LM_NOEXCEPT ON CACHE BOOL "" FORCE)

add_subdirectory(libjustlm)
add_subdirectory(anyproc)
add_subdirectory(cosched)
add_subdirectory(DPP)
add_subdirectory(fmt)

add_executable(discord_llama
    main.cpp
    config.hpp config.cpp
    utils.cpp utils.hpp
)
target_link_libraries(discord_llama PUBLIC dpp fmt pthread libjustlm anyproc ggml cosched sqlite3)

install(TARGETS discord_llama
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
