#ifndef WINDOWFRAME_H_
#define WINDOWFRAME_H_

#include <Common.h>

#include <map>
#include <string>
#include <vector>

namespace FRAME
{

class WindowFrame
{
public:
    virtual ~WindowFrame();

    static WindowFrame *GetWindowObject();

    GLFWwindow *GetGlfwWindow();

    int GetWindowFrameId();

    FRAMEID &GetWindowFrameObject();

    int GetPickupFrameId();

    FRAMEID &GetPickupFrameObject();

    int GetAppFrameId();

    FRAMEID &GetAppFrameObject();

public:
    int CreateGflwWindow(const std::string &str, int w, int h);

    int CreateGflwWindowFrame(const std::string &str, int w, int h);

public:
    int CreateDepthTexure(FRAMEID &depthFrame, int w, int h);

    int UpdateDepthTexure(FRAMEID &depthFrame, int w, int h);

    int CreateFloatFrame(FRAMEID &floatFrame, int w, int h);

    int UpdateFloatFrame(FRAMEID &floatFrame, int w, int h);

public:
    int CreateFrameBuffer(GLuint *id, int num);

    int CreateColorBuffer(GLuint &id, GLuint *buffer, int num, int w, int h, bool rgb);

    int CreateDepthBuffer(GLuint &id, GLuint *buffer, int num, int w, int h);

    int CreateDepth24Buffer(GLuint &id, GLuint *buffer, int num, int w, int h);

    int UpdateColorBuffer(GLuint &id, GLuint *buffer, int num, int w, int h, bool rgb);

    int UpdateDepthBuffer(GLuint &id, GLuint *buffer, int num, int w, int h);

    int UpdateDepth24Buffer(GLuint &id, GLuint *buffer, int num, int w, int h);

public:
    int CreateG_BufferFrame(GBFRAMEID &buffer, int w, int h);

    int UpdateG_BufferFrame(GBFRAMEID &buffer, int w, int h);

public:
    int CreateNoiseColorBuffer(GLuint &buffer, const std::vector<glm::vec3> &ssaoNoise);

public:
    void DeleteFrameBuffer(GLuint *buffer, int num);

    void DeleteFrameStruct(FRAMEID &frame);

    void DeleteColorBuffer(GLuint *buffer, int num);

    void DeleteDepthBuffer(GLuint *buffer, int num);

    void DeleteG_Buffer(GBFRAMEID &buffer);

public:
    void UpdateGflwWindowFrame(int w, int h);

private:
    void createWindowFraem(int w, int h, FRAMEID *frameType);

    void UpdateWindowFraem(int w, int h, FRAMEID *farem);

private:
    FRAMEID _windowFrame;

    FRAMEID _pickupFrame;

    FRAMEID _appBufferFrame;

    GLFWwindow *_glfwWindows = nullptr;
};

}    // namespace FRAME
#endif    // !WINDOWFRAME_H_
