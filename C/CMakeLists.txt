cmake_minimum_required(VERSION 3.23)

project(CSolver)

add_executable(cldc)

target_sources(cldc
    PRIVATE
        main.c
)

target_link_libraries(cldc m)

target_compile_options(cldc PRIVATE -pg -g -O3 -march=native -fopt-info-vec-optimized -fopt-info-vec-missed)
target_link_options(cldc PRIVATE -pg)
