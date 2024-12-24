cmake_minimum_required(VERSION 3.17)
project(MFCTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
set(CMAKE_MFC_FLAG 1)
find_package(MFC)
if (NOT MFC_FOUND)
    message(FATAL_ERROR "MFC is not found, please check MFC installation")
endif ()

#glad
set(GLAD_DIR 3rd/glad)
list(APPEND INCLUDE_DIR ${GLAD_DIR}/include)

#GLFW
set(GLFW_DIR 3rd/glfw)
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
#add_subdirectory(${GLFW_DIR} glfw_binary EXCLUDE_FROM_ALL)
list(APPEND INCLUDE_DIR ${GLFW_DIR}/include)
#list(APPEND INCLUDE_DIR ${GLFW_DIR}/deps)
list(APPEND LINK_LIBS glfw)

add_executable(MFCTest WIN32)
target_sources(MFCTest PRIVATE
        main.cpp
)

#set_target_properties(MFCTest PROPERTIES LINK_FLAGS "/SUBSYSTEM:WINDOWS")
target_compile_definitions(MFCTest PRIVATE
        -DWIN32
        -D_DEBUG
        -D_WINDOWS
        -D_VC80_UPGRADE=0x0600
        -D_UNICODE
        -DUNICODE
        -D_AFXDLL
)
target_link_options(MFCTest PRIVATE
        /ENTRY:wWinMainCRTStartup
)