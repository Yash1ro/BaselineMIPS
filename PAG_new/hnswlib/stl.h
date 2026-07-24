



























#ifndef ANKERL_STL_H
#define ANKERL_STL_H

#include <array>            
#include <cstdint>          
#include <cstring>          
#include <functional>       
#include <initializer_list> 
#include <iterator>         
#include <limits>           
#include <memory>           
#include <optional>         
#include <stdexcept>        
#include <string>           
#include <string_view>      
#include <tuple>            
#include <type_traits>      
#include <utility>          
#include <vector>           






#if defined __MINGW64__ && defined __GNUC__ && __GNUC__ >= 13 && !defined _REENTRANT


#    ifndef _WIN32_WINNT
#        error "_WIN32_WINNT not defined"
#    endif
#    if _WIN32_WINNT < 0x600
#        define ANKERL_MEMORY_RESOURCE_IS_BAD() 1 
#    endif
#endif
#ifndef ANKERL_MEMORY_RESOURCE_IS_BAD
#    define ANKERL_MEMORY_RESOURCE_IS_BAD() 0 
#endif

#if defined(__has_include) && !defined(ANKERL_UNORDERED_DENSE_DISABLE_PMR)
#    if __has_include(<memory_resource>) && !ANKERL_MEMORY_RESOURCE_IS_BAD()
#        define ANKERL_UNORDERED_DENSE_PMR std::pmr 
#        include <memory_resource>                  
#    elif __has_include(<experimental/memory_resource>)
#        define ANKERL_UNORDERED_DENSE_PMR std::experimental::pmr 
#        include <experimental/memory_resource>                   
#    endif
#endif

#if defined(_MSC_VER) && defined(_M_X64)
#    include <intrin.h>
#    pragma intrinsic(_umul128)
#endif

#endif
