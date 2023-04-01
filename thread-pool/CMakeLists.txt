cmake_minimum_required (VERSION 3.5)
project (threadpool)
 
add_library(threadpool INTERFACE)
target_include_directories(threadpool INTERFACE include)
