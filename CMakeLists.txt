cmake_minimum_required(VERSION 3.20)
project(keel_signal VERSION 0.1.0 LANGUAGES CXX)

option(KEEL_SIGNAL_NO_EXCEPTIONS_RTTI "Compile keel_signal without C++ exceptions and RTTI" OFF)
option(KEEL_SIGNAL_BUILD_TESTS "Build keel_signal unit tests and header purity checks" ${PROJECT_IS_TOP_LEVEL})

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
endif()
if(NOT TARGET keel::keel)
    include(FetchContent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../keel/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL "${CMAKE_CURRENT_SOURCE_DIR}/../keel" CACHE PATH "Local keel checkout")
    endif()
    if(KEEL_SIGNAL_NO_EXCEPTIONS_RTTI)
        set(KEEL_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel
        GIT_REPOSITORY https://github.com/keelcpp/keel.git
        GIT_TAG        4b37783925fe7af7a64daaff90f268df267fbbf3
    )
    FetchContent_MakeAvailable(keel)
endif()

if(NOT TARGET keel::fft)
    find_package(keel_fft QUIET CONFIG)
    if(TARGET keel::keel_fft AND NOT TARGET keel::fft)
        add_library(keel::fft ALIAS keel::keel_fft)
    endif()
endif()
if(NOT TARGET keel::fft)
    include(FetchContent)
    set(KEEL_FFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../fft/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_FFT)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_FFT "${CMAKE_CURRENT_SOURCE_DIR}/../fft" CACHE PATH "Local keel_fft checkout")
    endif()
    if(KEEL_SIGNAL_NO_EXCEPTIONS_RTTI)
        set(KEEL_FFT_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel_fft
        GIT_REPOSITORY https://github.com/keelcpp/fft.git
        GIT_TAG        8f27754ba6e5671439d97d3847479bdbf7ccfee7
    )
    FetchContent_MakeAvailable(keel_fft)
endif()

if(NOT TARGET keel::poly)
    find_package(keel_poly QUIET CONFIG)
    if(TARGET keel::keel_poly AND NOT TARGET keel::poly)
        add_library(keel::poly ALIAS keel::keel_poly)
    endif()
endif()
if(NOT TARGET keel::poly)
    include(FetchContent)
    set(KEEL_POLY_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../poly/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_POLY)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_POLY "${CMAKE_CURRENT_SOURCE_DIR}/../poly" CACHE PATH "Local keel_poly checkout")
    endif()
    if(KEEL_SIGNAL_NO_EXCEPTIONS_RTTI)
        set(KEEL_POLY_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel_poly
        GIT_REPOSITORY https://github.com/keelcpp/poly.git
        GIT_TAG        49f000d390c73b0cb426f41f11465aaf0f24e82a
    )
    FetchContent_MakeAvailable(keel_poly)
endif()

add_library(keel_signal STATIC
    src/windows.cpp
    src/conv.cpp
    src/filter.cpp
    src/transforms.cpp
    src/dct.cpp
    src/fwht.cpp
    src/hilbert.cpp
    src/smoothing.cpp
    src/findpeaks.cpp
    src/spectral.cpp
    src/iir_design.cpp
    src/multirate.cpp
    src/fir_design.cpp
    src/conversions.cpp
    src/modeling.cpp
    src/spectral_metrics.cpp
    src/waveforms.cpp
    src/pulse_roi.cpp
)
add_library(keel::signal ALIAS keel_signal)

target_link_libraries(keel_signal
    PUBLIC  keel::keel
    PRIVATE keel::fft keel::poly
)

target_compile_features(keel_signal PUBLIC cxx_std_20)
set_target_properties(keel_signal PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(keel_signal PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(keel_signal PRIVATE /W4 /utf-8)
    if(KEEL_SIGNAL_NO_EXCEPTIONS_RTTI)
        string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        target_compile_options(keel_signal PUBLIC /EHs-c- /GR-)
        target_compile_definitions(keel_signal PUBLIC _HAS_EXCEPTIONS=0)
    endif()
else()
    target_compile_options(keel_signal PRIVATE -Wall -Wextra -Wpedantic)
    if(KEEL_SIGNAL_NO_EXCEPTIONS_RTTI)
        target_compile_options(keel_signal PUBLIC -fno-exceptions -fno-rtti)
    endif()
endif()

if(COMMAND keel_enforce_guidelines)
    keel_enforce_guidelines(keel_signal
        SCOPE ALL
        BAN_RAW_INTS ALL
        BAN_RAW_FLOATS ALL
        HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include"
        STANDALONE_HEADERS ${KEEL_SIGNAL_BUILD_TESTS}
        TEST_NAME test_guidelines_headers
        WARNINGS OFF
    )
endif()

include(CMakePackageConfigHelpers)
install(TARGETS keel_signal EXPORT keel_signalTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/keel DESTINATION include)
install(EXPORT keel_signalTargets
    FILE keel_signalTargets.cmake
    NAMESPACE keel::
    DESTINATION lib/cmake/keel_signal
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/keel_signalConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_signalConfig.cmake"
    INSTALL_DESTINATION lib/cmake/keel_signal
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/keel_signalConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/keel_signalConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_signalConfigVersion.cmake"
    DESTINATION lib/cmake/keel_signal
)

if(KEEL_SIGNAL_BUILD_TESTS)
    enable_testing()

    add_executable(test_signal tests/test_signal.cpp)
    target_link_libraries(test_signal PRIVATE keel::signal)
    if(MSVC)
        target_compile_options(test_signal PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_signal PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if(COMMAND keel_enforce_guidelines)
        keel_enforce_guidelines(test_signal
            SCOPE ALL
            BAN_RAW_INTS ALL
            BAN_RAW_FLOATS ALL
            WARNINGS OFF
        )
    endif()
    add_test(NAME test_signal COMMAND test_signal)

    get_target_property(KEEL_CORE_INC_DIRS keel::keel INTERFACE_INCLUDE_DIRECTORIES)
    set(KEEL_CORE_INC_CLEAN "")
    foreach(IDIR IN LISTS KEEL_CORE_INC_DIRS)
        string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" IDIR_STRIPPED "${IDIR}")
        if(NOT IDIR_STRIPPED MATCHES "\\$<INSTALL_INTERFACE:" AND EXISTS "${IDIR_STRIPPED}")
            list(APPEND KEEL_CORE_INC_CLEAN "${IDIR_STRIPPED}")
        endif()
    endforeach()
    string(REPLACE ";" "|" KEEL_DEP_INC_JOINED "${KEEL_CORE_INC_CLEAN}")
    string(REPLACE ";" "|" KEEL_STD_INC_JOINED "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    add_test(
        NAME test_header_purity
        COMMAND ${CMAKE_COMMAND}
            "-DKEEL_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-DKEEL_DEP_INC_DIRS=${KEEL_DEP_INC_JOINED}"
            "-DCOMPILER_PATH=${CMAKE_CXX_COMPILER}"
            "-DCOMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
            "-DSTD_INC_DIRS=${KEEL_STD_INC_JOINED}"
            "-DCMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckHeaderPurity.cmake"
    )
endif()
