cmake_minimum_required(VERSION 3.15...3.26)


SET(CMAKE_Fortran_FLAGS "-fbounds-check -fbacktrace -fdiagnostics-color=auto")

set(CMAKE_VERBOSE_MAKEFILE on)
project(pyfm2d)
add_subdirectory(src/fm2d)
