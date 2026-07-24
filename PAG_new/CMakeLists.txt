
cmake_minimum_required(VERSION 3.10)
project(hnsw_lib_project CXX)



set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)


set(SOURCE_EXE main.cpp)
set(SOURCE_LIB pag.cpp)




set(COMMON_COMPILE_FLAGS
    -Ofast
    -march=native
    -fPIC
    -w
)



set(AVX512_EXT_FLAGS
    -mavx512f
    -mavx512dq
    -mavx512cd
    -mavx512bw
    -mavx512vl
    -mavx512vnni
)



find_package(OpenMP REQUIRED)


add_library(pag STATIC ${SOURCE_LIB})


target_compile_options(pag PRIVATE
    ${COMMON_COMPILE_FLAGS}
    ${AVX512_EXT_FLAGS}
)


if(OpenMP_CXX_FOUND)
    target_link_libraries(pag PRIVATE OpenMP::OpenMP_CXX)

    target_compile_options(pag PRIVATE ${OpenMP_CXX_FLAGS})
endif()


add_executable(PEOs ${SOURCE_EXE})


target_compile_options(PEOs PRIVATE
    ${COMMON_COMPILE_FLAGS}
    ${AVX512_EXT_FLAGS}
)


target_link_libraries(PEOs PRIVATE pag)

if(OpenMP_CXX_FOUND)
    target_link_libraries(PEOs PRIVATE OpenMP::OpenMP_CXX)

    target_compile_options(PEOs PRIVATE ${OpenMP_CXX_FLAGS})
endif()
