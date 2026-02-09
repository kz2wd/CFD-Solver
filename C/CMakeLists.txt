cmake_minimum_required(VERSION 3.23)

project(CSolver)

add_executable(cldc)

target_sources(cldc
    PRIVATE
        main.c
)

target_link_libraries(cldc m)
