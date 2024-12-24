#ifndef STRINGHELPER_H
#define STRINGHELPER_H
#include <string>
#include <vector>

namespace COMMON
{
std::string Ansi2Utf8(const char *txt, int length);

std::string Utf82gbk(const char *strutf);

bool isGBK(const char *data, int len);

bool isUtf8(const char *data, int len);

std::string ToUtf8(const char *data);

std::string ToGBK(const char *data);

std::string GetFileName(const char *path);

std::string ReserveDecimals(const float &val, int num,bool flag = false);

bool GetSuffixPath(const char *directory, const char *format, std::vector<std::string> &out);

}    // namespace COMMON

#endif    // !STRINGHELPER_H