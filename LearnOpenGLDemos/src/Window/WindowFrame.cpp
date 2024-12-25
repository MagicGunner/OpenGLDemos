#include "WindowFrame.h"

namespace FRAME
{

static void glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

WindowFrame::~WindowFrame()
{
    DeleteFrameStruct(_windowFrame);
    DeleteFrameStruct(_pickupFrame);
    DeleteFrameStruct(_appBufferFrame);
}

WindowFrame *WindowFrame::GetWindowObject()
{
    static WindowFrame frame;
    return &frame;
}

GLFWwindow *WindowFrame::GetGlfwWindow()
{
    return _glfwWindows;
}

int WindowFrame::GetWindowFrameId()
{
    return _windowFrame.Frame;
}

FRAMEID &WindowFrame::GetWindowFrameObject()
{
    return _windowFrame;
}

int WindowFrame::GetPickupFrameId()
{
    return _pickupFrame.Frame;
}

FRAMEID &WindowFrame::GetPickupFrameObject()
{
    return _pickupFrame;
}

int WindowFrame::GetAppFrameId()
{
    return _appBufferFrame.Frame;
}

FRAMEID &WindowFrame::GetAppFrameObject()
{
    return _appBufferFrame;
}

int WindowFrame::CreateGflwWindow(const std::string &str, int w, int h)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 16);
    // glfwWindowHint(GLFW_DEPTH_BITS, 32);

    _glfwWindows = glfwCreateWindow(w, h, str.c_str(), nullptr, nullptr);
    if (_glfwWindows == nullptr) {
        return 1;
    }

    glfwMakeContextCurrent(_glfwWindows);

    // 以将渲染速率与显示器的刷新率同步。参数1表示启用垂直同步，0表示禁用。
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        return 1;
    }

    return 0;
}

int WindowFrame::CreateGflwWindowFrame(const std::string &str, int w, int h)
{
    if (CreateGflwWindow(str, w, h)) {
        return 1;
    }

    createWindowFraem(w, h, &_windowFrame);
    createWindowFraem(w, h, &_pickupFrame);
    createWindowFraem(w, h, &_appBufferFrame);

    return 0;
}

int WindowFrame::CreateDepthTexure(FRAMEID &depthFrame, int w, int h)
{
    glGenFramebuffers(1, &depthFrame.Frame);
    glGenTextures(1, &depthFrame.Texture);
    glBindTexture(GL_TEXTURE_2D, depthFrame.Texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    GLfloat borderColor[] = {1.0, 1.0, 1.0, 1.0};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    glBindFramebuffer(GL_FRAMEBUFFER, depthFrame.Frame);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthFrame.Texture, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

int WindowFrame::UpdateDepthTexure(FRAMEID &depthFrame, int w, int h)
{
    glBindTexture(GL_TEXTURE_2D, depthFrame.Texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    GLfloat borderColor[] = {1.0, 1.0, 1.0, 1.0};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    glBindFramebuffer(GL_FRAMEBUFFER, depthFrame.Frame);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthFrame.Texture, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

int WindowFrame::CreateFloatFrame(FRAMEID &floatFrame, int w, int h)
{
    glGenFramebuffers(1, &floatFrame.Frame);

    glGenTextures(1, &floatFrame.Texture);
    glBindTexture(GL_TEXTURE_2D, floatFrame.Texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenRenderbuffers(1, &floatFrame.Depth);
    glBindRenderbuffer(GL_RENDERBUFFER, floatFrame.Depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);

    glBindFramebuffer(GL_FRAMEBUFFER, floatFrame.Frame);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, floatFrame.Texture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, floatFrame.Depth);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

int WindowFrame::UpdateFloatFrame(FRAMEID &floatFrame, int w, int h)
{
    glBindTexture(GL_TEXTURE_2D, floatFrame.Texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindRenderbuffer(GL_RENDERBUFFER, floatFrame.Depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);

    glBindFramebuffer(GL_FRAMEBUFFER, floatFrame.Frame);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, floatFrame.Texture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, floatFrame.Depth);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

int WindowFrame::CreateFrameBuffer(GLuint *id, int num)
{
    glGenFramebuffers(num, id);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return 0;
}

int WindowFrame::CreateColorBuffer(GLuint &id, GLuint *buffer, int num, int w, int h, bool rgb)
{
    glBindFramebuffer(GL_FRAMEBUFFER, id);

    glGenTextures(num, buffer);
    for (unsigned int i = 0; i < num; i++) {
        glBindTexture(GL_TEXTURE_2D, buffer[i]);
        if (rgb) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        } else {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
        }

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, buffer[i], 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

int WindowFrame::CreateDepthBuffer(GLuint &id, GLuint *buffer, int num, int w, int h)
{
    glBindFramebuffer(GL_FRAMEBUFFER, id);

    glGenRenderbuffers(num, buffer);
    for (unsigned int i = 0; i < num; i++) {
        glBindRenderbuffer(GL_RENDERBUFFER, buffer[i]);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buffer[i]);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

int WindowFrame::CreateDepth24Buffer(GLuint &id, GLuint *buffer, int num, int w, int h)
{
    glBindFramebuffer(GL_FRAMEBUFFER, id);

    glGenRenderbuffers(num, buffer);
    for (unsigned int i = 0; i < num; i++) {
        glBindRenderbuffer(GL_RENDERBUFFER, buffer[i]);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buffer[i]);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

int WindowFrame::UpdateColorBuffer(GLuint &id, GLuint *buffer, int num, int w, int h, bool rgb)
{
    glBindFramebuffer(GL_FRAMEBUFFER, id);

    for (unsigned int i = 0; i < num; i++) {
        glBindTexture(GL_TEXTURE_2D, buffer[i]);
        if (rgb) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        } else {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, buffer[i], 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

int WindowFrame::UpdateDepthBuffer(GLuint &id, GLuint *buffer, int num, int w, int h)
{
    glBindFramebuffer(GL_FRAMEBUFFER, id);

    for (unsigned int i = 0; i < num; i++) {
        glBindRenderbuffer(GL_RENDERBUFFER, buffer[i]);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buffer[i]);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

int WindowFrame::UpdateDepth24Buffer(GLuint &id, GLuint *buffer, int num, int w, int h)
{
    glBindFramebuffer(GL_FRAMEBUFFER, id);

    for (unsigned int i = 0; i < num; i++) {
        glBindRenderbuffer(GL_RENDERBUFFER, buffer[i]);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buffer[i]);
    }

    return 0;
}

int WindowFrame::CreateG_BufferFrame(GBFRAMEID &buffer, int w, int h)
{
    glGenFramebuffers(1, &buffer.Frame);
    glBindFramebuffer(GL_FRAMEBUFFER, buffer.Frame);

    glGenTextures(1, &buffer.Position);
    glBindTexture(GL_TEXTURE_2D, buffer.Position);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buffer.Position, 0);

    glGenTextures(1, &buffer.Normal);
    glBindTexture(GL_TEXTURE_2D, buffer.Normal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, buffer.Normal, 0);

    glGenTextures(1, &buffer.AlbedoSpec);
    glBindTexture(GL_TEXTURE_2D, buffer.AlbedoSpec);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, buffer.AlbedoSpec, 0);

    unsigned int attachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, attachments);

    glGenRenderbuffers(1, &buffer.Depth);
    glBindRenderbuffer(GL_RENDERBUFFER, buffer.Depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buffer.Depth);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

int WindowFrame::UpdateG_BufferFrame(GBFRAMEID &buffer, int w, int h)
{
    glBindFramebuffer(GL_FRAMEBUFFER, buffer.Frame);

    glBindTexture(GL_TEXTURE_2D, buffer.Position);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buffer.Position, 0);

    glBindTexture(GL_TEXTURE_2D, buffer.Normal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, buffer.Normal, 0);

    glBindTexture(GL_TEXTURE_2D, buffer.AlbedoSpec);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, buffer.AlbedoSpec, 0);

    unsigned int attachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, attachments);

    glBindRenderbuffer(GL_RENDERBUFFER, buffer.Depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buffer.Depth);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

int WindowFrame::CreateNoiseColorBuffer(GLuint &buffer, const std::vector<glm::vec3> &ssaoNoise)
{
    glGenTextures(1, &buffer);
    glBindTexture(GL_TEXTURE_2D, buffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    return 0;
}

void WindowFrame::DeleteFrameBuffer(GLuint *buffer, int num)
{
    glDeleteFramebuffers(num, buffer);
}

void WindowFrame::DeleteFrameStruct(FRAMEID &frame)
{
    glDeleteTextures(1, &frame.Texture);
    glDeleteFramebuffers(1, &frame.Frame);
    glDeleteRenderbuffers(1, &frame.Depth);
}

void WindowFrame::DeleteColorBuffer(GLuint *buffer, int num)
{
    glDeleteTextures(num, buffer);
}

void WindowFrame::DeleteDepthBuffer(GLuint *buffer, int num)
{
    glDeleteRenderbuffers(num, buffer);
}

void WindowFrame::DeleteG_Buffer(GBFRAMEID &buffer)
{
    glDeleteFramebuffers(1, &buffer.Frame);
    glDeleteTextures(1, &buffer.Position);
    glDeleteTextures(1, &buffer.Normal);
    glDeleteTextures(1, &buffer.AlbedoSpec);
    glDeleteRenderbuffers(1, &buffer.Depth);
}

void WindowFrame::UpdateGflwWindowFrame(int w, int h)
{
    UpdateWindowFraem(w, h, &_windowFrame);
    UpdateWindowFraem(w, h, &_pickupFrame);
    UpdateWindowFraem(w, h, &_appBufferFrame);
}

void WindowFrame::createWindowFraem(int w, int h, FRAMEID *frameType)
{
    glGenFramebuffers(1, &frameType->Frame);
    glBindFramebuffer(GL_FRAMEBUFFER, frameType->Frame);

    glGenTextures(1, &frameType->Texture);
    glBindTexture(GL_TEXTURE_2D, frameType->Texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frameType->Texture, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenRenderbuffers(1, &frameType->Depth);
    glBindRenderbuffer(GL_RENDERBUFFER, frameType->Depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, frameType->Depth);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void WindowFrame::UpdateWindowFraem(int w, int h, FRAMEID *farem)
{
    glBindFramebuffer(GL_FRAMEBUFFER, farem->Frame);

    glBindTexture(GL_TEXTURE_2D, farem->Texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, farem->Texture, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindRenderbuffer(GL_RENDERBUFFER, farem->Depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, farem->Depth);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

}    // namespace FRAME
