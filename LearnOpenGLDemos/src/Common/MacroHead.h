#ifndef MACROHEAD_H_
#define MACROHEAD_H_
#include "glm/glm.hpp"

#include "GladGLfw.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif    // !PI

#ifndef MAXBONE
#define MAXBONE 4
#endif    // !MAXBONE

#ifndef IDWC
#define IDWC (2)
#endif    // !IDWC

#ifndef EPSINON
#define EPSINON (0.00000001f)
#endif    // !EPSINON

#ifndef SDelete
#define SDelete(ptr)          \
    do {                      \
        if (nullptr != ptr) { \
            delete ptr;       \
            ptr = nullptr;    \
        }                     \
    } while (0)
#endif

#ifndef SDeleteArray
#define SDeleteArray(ptr)     \
    do {                      \
        if (nullptr != ptr) { \
            delete[] ptr;     \
            ptr = nullptr;    \
        }                     \
    } while (0)
#endif

#ifndef NULLVABO
#define NULLVABO (0)
#endif    // !NULLVBO

#ifndef SDeleteVBO
#define SDeleteVBO(VBO)               \
    do {                              \
        if (VBO != NULLVABO) {        \
            glDeleteBuffers(1, &VBO); \
            VBO = NULLVABO;           \
        }                             \
    } while (0)
#endif

#ifndef SDeleteVAO
#define SDeleteVAO(VAO)                    \
    do {                                   \
        if (VAO != NULLVABO) {             \
            glDeleteVertexArrays(1, &VAO); \
            VAO = NULLVABO;                \
        }                                  \
    } while (0)
#endif

#ifndef SDeleteVABO
#define SDeleteVABO(VAO, VBO)              \
    do {                                   \
        if (VBO != NULLVABO) {             \
            glDeleteBuffers(1, &VBO);      \
            VBO = NULLVABO;                \
        }                                  \
        if (VAO != NULLVABO) {             \
            glDeleteVertexArrays(1, &VAO); \
            VAO = NULLVABO;                \
        }                                  \
    } while (0)
#endif

enum class KeyType : int
{
    Wkey = 0,
    Skey,
    Dkey,
    Akey,
    Qkey,
    Ekey
};

enum class VERTYPE : int
{
    VERTEX = 0,
    NORMAL,
    TEXURE,
    COLORS,
    INSTAN
};

enum class DRAWTYPE
{
    Points,
    LineStrip,
    LineLoop,
    Lines,
    Triangles,
    TriangleStrip,
    TriangleFan
};

enum class GAMESTATE
{
    GAME_ACTIVE,
    GAME_MENU,
    GAME_WIN
};

enum class GAMEDIR
{
    UP,
    RIGHT,
    DOWN,
    LEFT
};

struct PARTICLE {
    glm::vec2 Position, Velocity;
    glm::vec4 Color;
    GLfloat Life;

    PARTICLE()
        : Position(0.0f)
        , Velocity(0.0f)
        , Color(1.0f)
        , Life(0.0f)
    {
    }
};

const glm::vec3 PickRgbList[] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 0.0}, {1.0, 0.0, 1.0},
                                 {0.0, 1.0, 1.0}, {0.5, 0.5, 0.5}, {1.0, 0.5, 0.0}, {0.5, 1.0, 0.0}, {0.0, 0.5, 1.0},
                                 {0.5, 0.0, 1.0}, {1.0, 0.5, 0.5}, {0.5, 1.0, 0.5}, {0.5, 0.5, 1.0}, {1.0, 0.0, 0.5},
                                 {0.0, 1.0, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}, {0.7, 0.3, 0.2}, {0.2, 0.3, 0.7},
                                 {0.1, 0.1, 0.1}, {0.1, 0.2, 0.2}, {0.1, 0.3, 0.3}, {0.1, 0.4, 0.4}, {0.1, 0.5, 0.5}};

#endif    // !MACROHEAD_H_
