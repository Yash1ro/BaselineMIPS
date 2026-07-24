# 使用现代 CMake 版本，推荐 3.10 及以上
cmake_minimum_required(VERSION 3.10) 
project(hnsw_lib_project CXX)

## --- 1. 标准和扩展配置 ---
# 设置 C++ 标准为 C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 关闭 GNU 扩展，确保代码更符合标准 C++ 规范
set(CMAKE_CXX_EXTENSIONS OFF) 

# 源文件定义
set(SOURCE_EXE main.cpp)
set(SOURCE_LIB peos.cpp)

## --- 2. 编译标志和优化 ---

# 统一优化和架构标志
set(COMMON_COMPILE_FLAGS 
    -Ofast             # 积极优化
    -march=native      # 启用当前CPU支持的所有指令集（包括基础AVX/AVX512）
    -fPIC              # 生成位置无关代码 (用于共享库)
    -w                 # 抑制所有警告
)

# 强制添加完整的 AVX-512 扩展标志
# 这是解决特定非标准 Intrinsic (_mm512_extract_epi64) 声明缺失的关键
set(AVX512_EXT_FLAGS 
    -mavx512f 
    -mavx512dq 
    -mavx512cd 
    -mavx512bw 
    -mavx512vl
    -mavx512vnni
)

## --- 3. OpenMP 配置 ---
# 查找 OpenMP 包
find_package(OpenMP REQUIRED)

## --- 4. 库 (Library) 配置：peos ---
add_library(peos STATIC ${SOURCE_LIB})

# 应用编译选项：通用优化 + AVX-512 扩展
target_compile_options(peos PRIVATE 
    ${COMMON_COMPILE_FLAGS} 
    ${AVX512_EXT_FLAGS}
)

# 链接 OpenMP 库和设置编译标志
if(OpenMP_CXX_FOUND)
    target_link_libraries(peos PRIVATE OpenMP::OpenMP_CXX)
    # 将 -fopenmp 标志加入编译选项
    target_compile_options(peos PRIVATE ${OpenMP_CXX_FLAGS})
endif()

## --- 5. 可执行文件 (Executable) 配置：PEOs ---
add_executable(PEOs ${SOURCE_EXE})

# 应用编译选项：与 peos 库保持一致
target_compile_options(PEOs PRIVATE 
    ${COMMON_COMPILE_FLAGS} 
    ${AVX512_EXT_FLAGS}
)

# 链接 peos 库和 OpenMP 库
target_link_libraries(PEOs PRIVATE peos)

if(OpenMP_CXX_FOUND)
    target_link_libraries(PEOs PRIVATE OpenMP::OpenMP_CXX)
    # 将 -fopenmp 标志加入编译选项
    target_compile_options(PEOs PRIVATE ${OpenMP_CXX_FLAGS})
endif()
