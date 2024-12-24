#include "CaluHelper.h"

namespace COMMON
{

bool CalcuRgbZero(const glm::vec3 &rgb)
{
    if (rgb.x == 0.0f && rgb.y == 0.0f && rgb.z == 0.0f) {
        return true;
    }

    return false;
}

bool CalcuRgbDiff(const glm::vec3 &rgb, const glm::vec3 &rgb2)
{
    const float diff = 0.01f;
    if (fabs(rgb2.x - rgb.x) < diff && fabs(rgb2.y - rgb.y) < diff && fabs(rgb2.z - rgb.z) < diff) {
        return true;
    }

    return false;
}

void UcharConversionRGB(unsigned char *pixel, glm::vec3 &rgb)
{
    rgb.x = static_cast<float>(pixel[0]) / 255.0f;
    rgb.y = static_cast<float>(pixel[1]) / 255.0f;
    rgb.z = static_cast<float>(pixel[2]) / 255.0f;
}

}    // namespace COMMON
