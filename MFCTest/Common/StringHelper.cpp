#include "StringHelper.h"
#include "MacroHead.h"
#include <filesystem>
#include <io.h>
#include <windows.h>

namespace COMMON
{

std::string Ansi2Utf8(const char *txt, int length)
{
    int nwLen = ::MultiByteToWideChar(CP_ACP, 0, txt, -1, NULL, 0);
    wchar_t *pwBuf = new wchar_t[nwLen + 1];
    memset(pwBuf, 0, sizeof(wchar_t) * (nwLen + 1));
    ::MultiByteToWideChar(CP_ACP, 0, txt, length, pwBuf, nwLen);
    int nLen = ::WideCharToMultiByte(CP_UTF8, 0, pwBuf, -1, NULL, NULL, NULL, NULL);
    char *pBuf = new char[nLen + 1];
    memset(pBuf, 0, sizeof(char) * (nLen + 1));
    ::WideCharToMultiByte(CP_UTF8, 0, pwBuf, nwLen, pBuf, nLen, NULL, NULL);
    std::string retStr(pBuf);
    SDeleteArray(pwBuf);
    SDeleteArray(pBuf);
    return retStr;
}

std::string Utf82gbk(const char *strutf)
{
    int size = MultiByteToWideChar(CP_UTF8, 0, strutf, -1, NULL, 0);
    WCHAR *strUnicode = new WCHAR[size];
    MultiByteToWideChar(CP_UTF8, 0, strutf, -1, strUnicode, size);

    int i = WideCharToMultiByte(CP_ACP, 0, strUnicode, -1, NULL, 0, NULL, NULL);
    char *strGBK = new char[i];
    WideCharToMultiByte(CP_ACP, 0, strUnicode, -1, strGBK, i, NULL, NULL);
    return strGBK;
}

bool isGBK(const char *data, int len)
{
    int i = 0;
    while (i < len) {
        if (data[i] <= 0x7f) {
            //编码小于等于127,只有一个字节的编码，兼容ASCII
            i++;
            continue;
        } else {
            //大于127的使用双字节编码
            if (data[i] >= 0x81 && data[i] <= 0xfe && data[i + 1] >= 0x40 && data[i + 1] <= 0xfe && data[i + 1] != 0xf7)
            {
                i += 2;
                continue;
            } else {
                return false;
            }
        }
    }
    return true;
}

int preNUm(unsigned char byte)
{
    unsigned char mask = 0x80;
    int num = 0;
    for (int i = 0; i < 8; i++) {
        if ((byte & mask) == mask) {
            mask = mask >> 1;
            num++;
        } else {
            break;
        }
    }
    return num;
}

bool isUtf8(const char *data, int len)
{
    int num = 0;
    int i = 0;
    while (i < len) {
        if ((data[i] & 0x80) == 0x00) {
            // 0XXX_XXXX
            i++;
            continue;
        } else if ((num = preNUm(data[i])) > 2) {
            // 110X_XXXX 10XX_XXXX
            // 1110_XXXX 10XX_XXXX 10XX_XXXX
            // 1111_0XXX 10XX_XXXX 10XX_XXXX 10XX_XXXX
            // 1111_10XX 10XX_XXXX 10XX_XXXX 10XX_XXXX 10XX_XXXX
            // 1111_110X 10XX_XXXX 10XX_XXXX 10XX_XXXX 10XX_XXXX 10XX_XXXX
            // preNUm() 返回首个字节8个bits中首�?0bit前面1bit的个数，该数量也是该字符所使用的字节数
            i++;
            for (int j = 0; j < num - 1; j++) {
                //判断后面num - 1 个字节是不是都是10开
                if ((data[i] & 0xc0) != 0x80) {
                    return false;
                }
                i++;
            }
        } else {
            //其他情况说明不是utf-8
            return false;
        }
    }
    return true;
}

std::string ToUtf8(const char *data)
{
    std::string str = data;
    if (!isUtf8(data, str.length())) {
        str = Ansi2Utf8(data, str.length());
    }
    return str;
}

std::string ToGBK(const char *data)
{
    std::string str = data;
    if (isUtf8(data, str.length())) {
        str = Utf82gbk(data);
    }
    return str;
}

std::string GetFileName(const char *path)
{
    std::string str(path);
    std::string::size_type iPos = str.find_last_of('\\') + 1;
    std::string filename = str.substr(iPos, str.length() - iPos);
    std::string name = filename.substr(0, filename.rfind("."));
    return name;
}

std::string ReserveDecimals(const float &val, int num, bool flag)
{
    std::string str = std::to_string(val);
    str = str.substr(0, str.find(".") + num + 1);
    if (flag) {
        str = str.substr(0, str.length() - 1);
    }
    return str;
}

bool GetSuffixPath(const char *directory, const char *format, std::vector<std::string> &out)
{
    intptr_t hFile = 0;
    struct _finddata_t fileinfo;
    std::string p;
    std::string hz(format);

    if ((hFile = _findfirst(p.assign(directory).append("//*" + hz).c_str(), &fileinfo)) != -1) {
        do {
            if ((fileinfo.attrib & _A_SUBDIR)) {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    GetSuffixPath(p.assign(directory).append("\\").append(fileinfo.name).c_str(), format, out);
            } else {
                out.push_back(p.assign(directory).append("\\").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);

        _findclose(hFile);
    }

    return true;
}

}    // namespace COMMON
