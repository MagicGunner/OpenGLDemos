cmake_minimum_required(VERSION 3.17)
project(CMakeExe LANGUAGES C CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 通用变量设置
set(GLAD_DIR external/glad)
set(GLFW_DIR external/glfw)
set(IMGUI_DIR external/imgui)

set(LINK_LIBS "")        # 初始化链接库列表
set(INCLUDE_DIRS "")     # 初始化包含目录列表

# 查找 OpenGL
find_package(OpenGL REQUIRED)
list(APPEND LINK_LIBS OpenGL::GL)

# 配置 GLAD
list(APPEND INCLUDE_DIRS ${GLAD_DIR}/include)
list(APPEND OPENGL_SOURCE ${GLAD_DIR}/src/glad.c)

# 配置 GLFW
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)

add_subdirectory(${GLFW_DIR} glfw_binary EXCLUDE_FROM_ALL)
list(APPEND INCLUDE_DIRS ${GLFW_DIR}/include ${GLFW_DIR}/deps)
list(APPEND LINK_LIBS glfw)

# 配置 ImGui
list(APPEND INCLUDE_DIRS
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
)
list(APPEND IMGUI_SOURCE
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
)

# 通用的 OpenGL 源文件
#list(APPEND OPENGL_SOURCE src/OpenGLDemo.txt)

# 配置 OpenGL_DEMO
#add_executable(OpenGL_DEMO
#        ${OPENGL_SOURCE}
#)
#target_include_directories(OpenGL_DEMO PUBLIC ${INCLUDE_DIRS})
#target_link_libraries(OpenGL_DEMO PUBLIC ${LINK_LIBS})

# 配置 IMGUI_DEMO
add_executable(IMGUI_DEMO
        ${OPENGL_SOURCE}
        ${IMGUI_SOURCE}
        src/main.cpp
)
target_include_directories(IMGUI_DEMO PUBLIC ${INCLUDE_DIRS})
target_link_libraries(IMGUI_DEMO PUBLIC ${LINK_LIBS})

# 打印调试信息
message(STATUS "Include directories: ${INCLUDE_DIRS}")
message(STATUS "Link libraries: ${LINK_LIBS}")
