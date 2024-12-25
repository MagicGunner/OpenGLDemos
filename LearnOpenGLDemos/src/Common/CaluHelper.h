#ifndef CALUHELPER_H_
#define CALUHELPER_H_

#include "Common.h"
#include <string>

namespace COMMON
{

bool CalcuRgbZero(const glm::vec3 &rgb);

bool CalcuRgbDiff(const glm::vec3 &rgb, const glm::vec3 &rgb2);

void UcharConversionRGB(unsigned char *pixel, glm::vec3 &rgb);

}    // namespace COMMON
#endif    // !CALUHELPER_H_
