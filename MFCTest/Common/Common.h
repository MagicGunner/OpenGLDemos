#ifndef COMMON_H_
#define COMMON_H_
#include "GladGLfw.h"
#include "MacroHead.h"
#include <string>

template <typename T>
bool Equal(T a, T b)
{
    return std::abs(a - b) < std::numeric_limits<T>::epsilon();
}

template <typename T>
struct Point3D {
    typedef T value_type;

    T x;
    T y;
    T z;
    int idx;

    Point3D(T ix = 0, T iy = 0, T iz = 0)
        : x(ix)
        , y(iy)
        , z(iz)
        , idx(-1)
    {
    }

    Point3D(const glm::vec3 &pt)
        : x(pt.x)
        , y(pt.y)
        , z(pt.z)
        , idx(-1)
    {
    }

    void operator=(const glm::vec3 &pt) const
    {
        x = pt.x;
        y = pt.y;
        z = pt.z;
        idx = -1;
    }

    bool operator<(const Point3D<T> &pt) const
    {
        if (!Equal(x, pt.x)) {
            return x < pt.x;
        }

        if (!Equal(y, pt.y)) {
            return y < pt.y;
        }

        if (!Equal(z, pt.z)) {
            return z < pt.z;
        }

        return false;
    }

    bool operator==(const Point3D<T> &pt) const
    {
        return (Equal(pt.x, x) && Equal(pt.y, y) && Equal(pt.z, z));
    }

    T operator[](size_t i) const
    {
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            default:
                return T(0);
        }
    }

    Point3D<T> operator+(const Point3D<T> &pt) const
    {
        return Point3D<T>(x + pt.x, y + pt.y, z + pt.z);
    }

    Point3D<T> operator-(const Point3D<T> &pt) const
    {
        return Point3D<T>(x - pt.x, y - pt.y, z - pt.z);
    }

    Point3D<T> operator/(const T &value) const
    {
        return Point3D<T>(x / value, y / value, z / value);
    }
};

typedef Point3D<float> POINTKDT;

template <typename T>
struct Point2D {
    T x;
    T y;

    Point2D(T ix = 0, T iy = 0)
        : x(ix)
        , y(iy)
    {
    }

    bool operator<(const Point2D<T> &pt) const
    {
        if (!Equal(x, pt.x)) {
            return x < pt.x;
        }

        if (!Equal(y, pt.y)) {
            return y < pt.y;
        }

        return false;
    }

    bool operator==(const Point2D<T> &pt) const
    {
        return (Equal(pt.x, x) && Equal(pt.y, y));
    }

    T operator[](size_t i) const
    {
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            default:
                return T(0);
        }
    }

    Point2D<T> operator+(const Point2D<T> &pt) const
    {
        return Point2D<T>(x + pt.x, y + pt.y);
    }

    Point2D<T> operator-(const Point2D<T> &pt) const
    {
        return Point2D<T>(x - pt.x, y - pt.y);
    }

    Point2D<T> operator/(const T &value) const
    {
        return Point2D<T>(x / value, y / value);
    }
};

typedef Point2D<int> POINT2D_INT;

typedef struct dims {

    int x = 0;
    int y = 0;
    int z = 0;
} DIMS;

typedef struct MeshPos {

    glm::vec3 Pt[4];
} MESHPOS;

typedef struct vertexColor {

    glm::vec3 VPos;
    glm::vec3 Clor;
} VERCOR;

typedef struct vertexNorColor {

    glm::vec3 VPos;
    glm::vec3 Nors;
    glm::vec3 Clor;
} VERNORCOR;

typedef struct vertexNorTex {

    glm::vec3 VPos;
    glm::vec3 Nors;
    glm::vec2 Texs;
} VERNORTEX;

typedef struct MeshVerTex {

    glm::vec3 VPos = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 Nors = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec2 Texs = glm::vec2(0.0f, 0.0f);
    glm::vec3 Tans = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 Bits = glm::vec3(0.0f, 0.0f, 0.0f);
    int Bones[MAXBONE];
    float Weights[MAXBONE];

} MESHVERTEX;

typedef struct glFont {

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    std::string Text;

    template <typename T>
    void operator=(const T &pos)
    {
        x = pos.x;
        y = pos.y;
        z = pos.z;
    }

} GLFONT;

typedef struct FrameID {

    GLuint Frame = -1;
    GLuint Depth = -1;
    GLuint Texture = -1;

} FRAMEID;

typedef struct GBFrameID {

    GLuint Frame = -1;
    GLuint Position = -1;
    GLuint Normal = -1;
    GLuint AlbedoSpec = -1;
    GLuint Depth = -1;

} GBFRAMEID;

typedef struct TexureId {

    GLuint Id = 0;
    std::string type;

} TEXUREID;

typedef struct AreaLight {

    bool twoSided = true;
    float yRotation = 0.0f;
    float intensity = 4.0f;
    glm::vec3 color = glm::vec3(0.0f);
    glm::vec3 offset = glm::vec3(0.0f);

} AREALIGHT;

#endif    // !COMMON_H_
