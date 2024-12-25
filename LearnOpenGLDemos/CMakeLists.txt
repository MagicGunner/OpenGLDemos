cmake_minimum_required(VERSION 3.30)
project(LearnOpenGLDemos LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

### 项目名称 ###
SET(ProjectName LearnOpenGLDemos)
PROJECT(${ProjectName})

# 版本号设置
IF (USE_DEFAULT_VERSION)
    SET(VERSION_MAJOR_MAJOR 1)
    SET(VERSION_MAJOR_MINOR 0)
    SET(VERSION_PATCH 0)
    SET(VERSION_BUILD 0)
ENDIF (USE_DEFAULT_VERSION)

# 通用变量设置
set(INCLUDE_DIRS "")     # 初始化包含目录列表
set(LINK_LIBS "")        # 初始化链接库列表
set(OPENGL_SOURCE "")


# 配置GLAD
set(GLAD_DIR 3rd/glad)
list(APPEND INCLUDE_DIRS ${GLAD_DIR}/include)
list(APPEND OPENGL_SOURCE ${GLAD_DIR}/src/glad.c)


# 配置GLFW
set(GLFW_DIR 3rd/glfw)
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
add_subdirectory(${GLFW_DIR} glfw_binary EXCLUDE_FROM_ALL)
list(APPEND INCLUDE_DIRS ${GLFW_DIR}/include ${GLFW_DIR}/deps)
list(APPEND LINK_LIBS glfw)

# 配置IMGui
set(IMGUI_DIR 3rd/imgui)
list(APPEND INCLUDE_DIRS ${IMGUI_DIR} ${IMGUI_DIR}/backends)
list(APPEND OPENGL_SOURCE
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
)


# 添加头文件
list(APPEND INCLUDE_DIRS
        # 通用库
        src/Common
        # 数学库
        3rd/glm
)

file(GLOB WINDOW_SOURCE
        src/Window/*.cpp
        src/Window/*.h)

file(GLOB COMMON_SOURCE
        src/Common/*.cpp
        src/Common/*.h)

list(APPEND OPENGL_SOURCE
        main.cpp
        ${WINDOW_SOURCE}
        ${COMMON_SOURCE}
)

add_executable(${ProjectName} ${OPENGL_SOURCE})

target_include_directories(${ProjectName} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${ProjectName} PUBLIC ${LINK_LIBS})

message(STATUS "INCLUDE_DIRS: ${INCLUDE_DIRS}")
message(STATUS "LINK_LIBS: ${LINK_LIBS}")
message(STATUS "ProjectName: ${ProjectName}")


