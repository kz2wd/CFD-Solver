cmake_minimum_required(VERSION 3.23)

project(CSolver)

add_executable(cldc)

target_sources(cldc
    PRIVATE
        main.c
)

target_link_libraries(cldc m)

target_compile_options(cldc PRIVATE -pg -g)
target_link_options(cldc PRIVATE -pg)
