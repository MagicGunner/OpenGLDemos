cmake_minimum_required(VERSION 3.17)
project(MFCTest LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_MFC_FLAG 1)
#find_package(MFC)
#if (NOT MFC_FOUND)
#    message(FATAL_ERROR "MFC is not found, please check MFC installation")
#endif ()

#GLAD
set(GLAD_DIR external/glad)
list(APPEND INCLUDE_DIR ${GLAD_DIR}/include)

#GLFW
set(GLFW_DIR external/glfw)
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
add_subdirectory(${GLFW_DIR} glfw_binary EXCLUDE_FROM_ALL)
list(APPEND INCLUDE_DIR ${GLFW_DIR}/include)
list(APPEND INCLUDE_DIR ${GLFW_DIR}/deps)
list(APPEND LINK_LIBS glfw)

#ImGui
set(IMGUI_DIR ./external/imgui)
list(APPEND IMGUI_INCLUDE_DIR ${IMGUI_DIR} ${IMGUI_DIR}/backends)

# vulkan
find_package(Vulkan REQUIRED)
list(APPEND LINK_LIBS "Vulkan::Vulkan")

list(APPEND OPENGL_SOURCE
        ${GLAD_DIR}/src/glad.c
        src/OpenGLDemo.cpp
)

list(APPEND IMGUI_SOURCE
        src/main.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
)

add_executable(MFCTest ${OPENGL_SOURCE})
target_include_directories(MFCTest PUBLIC ${INCLUDE_DIR})
target_link_libraries(MFCTest PUBLIC ${LINK_LIBS})

add_executable(IMGUI_DEMO ${IMGUI_SOURCE})
target_include_directories(IMGUI_DEMO PUBLIC ${IMGUI_INCLUDE_DIR})
target_link_libraries(IMGUI_DEMO PUBLIC ${LINK_LIBS})

#target_sources(MFCTest PRIVATE
#        src/main.cpp
#)

#set_target_properties(MFCTest PROPERTIES LINK_FLAGS "/SUBSYSTEM:WINDOWS")
#target_compile_definitions(MFCTest PRIVATE
#        -DWIN32
#        -D_DEBUG
#        -D_WINDOWS
#        -D_VC80_UPGRADE=0x0600
#        -D_UNICODE
#        -DUNICODE
#        -D_AFXDLL
#)
#target_link_options(MFCTest PRIVATE
#        /ENTRY:wWinMainCRTStartup
#)