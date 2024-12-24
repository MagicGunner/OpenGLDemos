#ifndef GRAPHICS_H_
#define GRAPHICS_H_
#include "Common.h"
#include "KdTree.h"
#include "MacroHead.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <set>
#include <string>
#include <vector>

namespace COMMON
{
/*******************************************************************************/
/*                          点线,图形计算                                      */
/*******************************************************************************/

/**
 * @brief 2D欧几里得距离
 *
 * @param pt1 第一个点
 * @param pt2 第二个点
 * @return 两点之间的欧几里得距离
 * @author wubo
 * @date 2024/05/22
 * @remark 1.2024/05/22 13:35 created by wubo
 */
template <typename T1, typename T2>
auto Get2DDistance(const T1 &pt1, const T2 &pt2)
{
    auto dx = pt2.x - pt1.x;
    auto dy = pt2.y - pt1.y;

    return std::sqrt(dx * dx + dy * dy);
}

/**
 * @brief 3D欧几里得距离
 *
 * @param pt1 第一个点
 * @param pt2 第二个点
 * @return 两点之间的欧几里得距离
 * @author wubo
 * @date 2024/05/22
 * @remark 1.2024/05/22 13:35 created by wubo
 */
template <typename T1, typename T2>
auto Get3DDistance(const T1 &pt1, const T2 &pt2)
{
    auto dx = pt2.x - pt1.x;
    auto dy = pt2.y - pt1.y;
    auto dz = pt2.z - pt1.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * @brief 点组中寻找当前点的最近点
 *
 * @param pt                    当前点
 * @param pts                   点数组
 * @param ptNum                 点的数量
 * @param outIdx                最近点的索引
 * @param outDis                最近点的距离
 * @param DistanceFunc          距离函数类型
 *
 * PS: typename DistanceFunc 函数指针,2D距离,3D距离
 * PS: std::numeric_limits<T3>::max()  返回当前类型的最大值 min()最小值
 *
 * @author wubo
 * @date 2024/05/22
 * @remark 1.2024/05/22 13:35 created by wubo
 */
template <typename T1, typename T2, typename T3, typename DistanceFunc>
void GetNearestPoint(const T1 &pt, const T2 *pts, int ptNum, int &outIdx, T3 &outDis, DistanceFunc func)
{
    int idx = -1;
    T3 dis = std::numeric_limits<T3>::max();

    for (int i = 0; i < ptNum; i++) {
        T3 tmp = func(pt, pts[i]);
        if (tmp < dis) {
            idx = i;
            dis = tmp;
        }
    }

    outIdx = idx;
    outDis = dis;
}

/**
 * @brief 线性插值
 *
 * @param pt1                   线端点1
 * @param pt2                   线端点2
 * @param int num               插值点数
 * @param std::vector<T> &out   出参
 * @author wubo
 * @date 2024/05/24
 * @remark 1.2024/05/24 15:35 created by wubo
 *
 * PS:保留起始点和结束点
 */
template <typename T>
void LinearInterpolation(const T &pt1, const T &pt2, int num, std::vector<T> &out)
{
    out.clear();
    float step = 1.0f / (num + 1);

    for (int i = 0; i <= num + 1; i++) {
        float t = step * i;
        T inter;
        inter = pt1 + (pt2 - pt1) * t;
        out.push_back(inter);
    }
}

/**
   点到直线的距离
*
* @param : lineS        //线段起始点
* @param : lineE        //线段结束点
* @param : pt           //要计算的点
* @return : double
* @author wubo
* @date 2024/05/28
* @remark 1.2024/05/28 15:35 created by wubo
*/
template <typename T, typename T2>
double GetPointLineDistance(const T &lineS, const T &lineE, const T2 &pt)
{
    // 直线的向量
    double A = lineE.y - lineS.y;
    double B = lineS.x - lineE.x;
    double C = lineE.x * lineS.y - lineS.x * lineE.y;

    // 计算距离
    double distance = std::abs(A * pt.x + B * pt.y + C) / std::sqrt(A * A + B * B);
    return distance;
}

/**
 * @brief 离散点
 *
 * @param pts                   点组
 * @param int num               点数
 * @param float step            步长
 * @param std::vector<T> &out   出参
 * @author wubo
 * @date 2024/05/24
 * @remark 1.2024/05/24 15:35 created by wubo
 */
template <typename T, typename DistanceFunc>
void DiscretePoints(const T *pts, int num, float step, std::vector<T> &out, DistanceFunc func)
{
    if (num <= 0 || step <= 0.0f) {
        return;
    }

    out.clear();
    out.push_back(pts[0]);

    float accumulated_distance = 0.0f;
    for (int i = 1; i < num; ++i) {
        float d = func(pts[i], pts[i - 1]);
        accumulated_distance += d;
        if (accumulated_distance >= step) {
            out.push_back(pts[i]);
            accumulated_distance = 0.0f;
        }
    }
}

/**
 * @brief 平面中多边形边界线上插入任意点
 *
 * @param bdypts                    //边界点
 * @param int num                   //边界点数
 * @param pt                        //要插入点
 * @param std::vector<T> &out       //出参:加入新点的边界点
 * @return
 * @author wubo
 * @date 2024/05/30
 * @remark 1.2024/05/30 14:35 created by wubo
 */
template <typename T>
void BoundaryInsertPoint(const T *bdypts, int num, const T &pt, std::vector<T> &out)
{
    int idx = 0;
    float minDis = std::numeric_limits<float>::max();

    for (int i = 1; i < num; i++) {

        std::vector<T> linear;
        int linePtNum = std::ceil(Get2DDistance<T, T>(bdypts[i - 1], bdypts[i]));
        LinearInterpolation<T>(bdypts[i - 1], bdypts[i], linePtNum, linear);

        for (int j = 0; j < linear.size(); j++) {
            float currDis = Get2DDistance<T, T>(linear[j], pt);
            if (currDis < minDis) {
                idx = i - 1;
                minDis = currDis;
            }
        }
    }

    for (int i = 0; i < num; i++) {
        out.push_back(bdypts[i]);
        if (i == idx) {
            out.push_back(pt);
        }
    }
}

/**
 * @brief 计算龙曲线分型
 *
 * @param const T &staPt            //线段开始点
 * @param const T &endPt            //线段结束点
 * @param int calcuNum              //分型次数
 * @param std::vector<T> &outPts    //出参
 * @return int
 * @author wubo
 * @date 2024/05/24
 * @remark 1.2024/05/24 15:35 created by wubo
 */
template <typename T>
void DragonShapedCurves(const T &staPt, const T &endPt, int calcuNum, std::vector<T> &outPts)
{
    outPts.push_back(staPt);
    outPts.push_back(endPt);

    if (calcuNum <= 0) {
        return;
    }

    // 计算中点和旋转
    T midPt = {(staPt.x + endPt.x) / 2, (staPt.y + endPt.y) / 2, 0.0f};
    double dx = endPt.y - staPt.y;
    double dy = staPt.x - endPt.x;
    T rotatePt = {midPt.x + dx, midPt.y + dy, 0.0f};

    // 递归生成龙形曲线的分型
    DragonShapedCurves(staPt, midPt, calcuNum - 1, outPts);
    DragonShapedCurves(endPt, midPt, calcuNum - 1, outPts);
    DragonShapedCurves(midPt, rotatePt, calcuNum - 1, outPts);
}

/*************************叉积***************************/

/**
 * @brief 计算两个二维向量的叉积
 *
 * @param p1 向量1
 * @param p2 向量2
 * @return 返回两个向量的叉积（标量）
 * @author wubo
 * @date 2024/05/24
 * @remark 1.2024/05/24 15:35 created by wubo
 */
template <typename T, typename T2>
float CrossProduct(const T &p1, const T2 &p2)
{
    return p1.x * p2.y - p1.y * p2.x;
}

/**
 * @brief 计算两个三维向量的叉积
 *
 * @param p1 向量1
 * @param p2 向量2
 * @param out 输出向量，存储叉积结果
 * @author wubo
 * @date 2024/05/24
 * @remark 1.2024/05/24 15:35 created by wubo
 */
template <typename T, typename T2, typename T3>
void CrossProduct(const T &p1, const T2 &p2, T3 &out)
{
    out.x = p1.y * p2.z - p1.z * p2.y;
    out.y = p1.z * p2.x - p1.x * p2.z;
    out.z = p1.x * p2.y - p1.y * p2.x;
}

/*************************点积***************************/

/**
 * @brief 计算两个向量的点积
 *
 * @param p1 向量1
 * @param p2 向量2
 * @return 返回两个向量的点积
 * @author wubo
 * @date 2024/05/24
 * @remark 1.2024/05/24 15:35 created by wubo
 *
 */
template <typename T, typename T2>
float DotProduct2D(const T &p1, const T2 &p2)
{
    return p1.x * p2.x + p1.y * p2.y;
}

/**
 * @brief 计算两个向量的点积
 *
 * @param p1 向量1
 * @param p2 向量2
 * @return 返回两个向量的点积
 * @author wubo
 * @date 2024/05/24
 * @remark 1.2024/05/24 15:35 created by wubo
 */
template <typename T, typename T2>
float DotProduct3D(const T &p1, const T2 &p2)
{
    return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

/*************************计算向量长度(模长)***************************/

/**
 * @brief 计算向量模长
 *
 * @param vec 向量
 * @return 返回向量模长
 * @author wubo
 * @date 2024/05/24
 * @remark 1.2024/05/24 15:35 created by wubo
 */
template <typename T>
float VectorLength2D(const T &vec)
{
    return sqrt(vec.x * vec.x + vec.y * vec.y);
}

template <typename T>
float VectorLength3D(const T &p1)
{
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

/*****************************向量归一化*****************************/

/**
 * @brief 向量归一化
 *
 * @param vec 向量
 * @param out 出参
 * @author wubo
 * @date 2024/05/24
 * @remark 1.2024/05/24 15:35 created by wubo
 */
template <typename T>
void NormalizeVector2D(const T &vec, T &out)
{
    float length = sqrt(vec.x * vec.x + vec.y * vec.y);
    if (length != 0.0f) {
        out.x /= length;
        out.y /= length;
    }
}

template <typename T>
void NormalizeVector3D(const T &vec, T &out)
{
    float length = sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    if (length != 0.0f) {
        out.x /= length;
        out.y /= length;
        out.z /= length;
    }
}

/*****************************向量角度计算*****************************/

/**
 * @brief 计算两个二维向量之间的夹角（以弧度表示）
 *
 * @param vec1 向量1
 * @param vec2 向量2
 * @return 向量之间的夹角（弧度）
 * @author wubo
 * @date 2024/05/24
 * @remark 1.2024/05/24 15:35 created by wubo
 */
template <typename T, typename T2>
float NormalizeVector2D(const T &vec, const T &vec2)
{
    float dot = DotProduct2D(vec1, vec2);
    float len1 = vectorLength2D(vec1);
    float len2 = vectorLength2D(vec2);
    return acos(dot / (len1 * len2));
}

template <typename T, typename T2>
float NormalizeVector3D(const T &vec, const T &vec2)
{
    float dot = DotProduct3D(vec1, vec2);
    float len1 = vectorLength3D(vec1);
    float len2 = vectorLength3D(vec2);
    return acos(dot / (len1 * len2));
}

/**
 * @brief 计算两点之间的极角
 *
 * @param a
 * @param b
 * @return double
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 15:59 created by wubo
 */
template <typename T>
double PolarAngle(const T &a, const T &b)
{
    return atan2(b.y - a.y, b.x - a.x);
}

/**
 * @brief 计算三角形的法线向量
 *
 * @param pt1 三角形的第一个顶点
 * @param pt2 三角形的第二个顶点
 * @param pt3 三角形的第三个顶点
 * @param out 存储法线向量的输出变量
 * @return void
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 15:59 created by wubo
 */
template <typename T, typename T2>
void CalcuTriangleNormal(const T &pt1, const T &pt2, const T &pt3, T2 &out)
{
    glm::vec3 v1(pt2.x - pt1.x, pt2.y - pt1.y, pt2.z - pt1.z);
    glm::vec3 v2(pt3.x - pt1.x, pt3.y - pt1.y, pt3.z - pt1.z);

    v1 = glm::normalize(v1);
    v2 = glm::normalize(v2);

    glm::vec3 tempNormal = glm::cross(v1, v2);
    glm::vec3 normal = glm::normalize(tempNormal);

    out.x = normal.x;
    out.y = normal.y;
    out.z = normal.z;
}

/**
 * @brief 计算平面两条线段的方向是否相同
 *
 * @param l1s                   第一条线段起始点
 * @param l1e                   第一条线段结束点
 * @param l2s                   第二条线段起始点
 * @param l2e                   第二条线段结束点
 * @return float cosine         余弦值
 * @author wubo
 * @date 2024/05/27
 * @remark 1.2024/05/27 10:35 created by wubo
 *
 * PS:余弦值接近1：则表示两条线段的方向相同
 * PS:余弦值接近-1：则表示两条线段的方向相反
 * PS:余弦值不接近1或-1，则表示两条线段的方向不同，可能存在一定的夹角。
 * PS < 0.0f,方向相反,  >0.0f 方向相同
 *
 */
template <typename T, typename T2>
float LineDirectionSame(const T &l1s, const T &l1e, const T2 &l2s, const T2 &l2e)
{
    float sx = l1e.x - l1s.x;
    float sy = l1e.y - l1s.y;
    float ex = l2e.x - l2s.x;
    float ey = l2e.y - l2s.y;

    double len = std::sqrt(sx * sx + sy * sy);
    double len2 = std::sqrt(ex * ex + ey * ey);

    double dot = sx * ex + sy * ey;
    double cosine = dot / (len * len2);

    return cosine;
}

/**
 * @brief 计算平面两条线段的交点
 *
 * @param s1p                   第一条线段起始点
 * @param e1p                   第一条线段结束点
 * @param s2p                   第二条线段起始点
 * @param e2p                   第二条线段结束点
 * @param out                   交点出参
 * @author wubo
 * @date 2024/05/27
 * @remark 1.2024/05/27 10:35 created by wubo
 *
 * PS:返回 0 存在交点, 返回 1 不存在交点或者共线
 *
 */
template <typename T, typename T2, typename T3>
int GetCrossPoint(const T &s1p, const T &e1p, const T2 &s2p, const T2 &e2p, T3 &out)
{
    auto d1 = e1p - s1p;
    auto d2 = e2p - s2p;
    auto delta = CrossProduct(d1, d2);

    if (delta == 0) {
        return 1;
    }

    auto s = s2p - s1p;
    auto t1 = CrossProduct(s, d2) / delta;
    auto t2 = CrossProduct(s, d1) / delta;

    if (t1 < 0 || t1 > 1 || t2 < 0 || t2 > 1) {
        return 1;
    }

    out.x = s1p.x + t1 * d1.x;
    out.y = s1p.y + t1 * d1.y;

    return 0;
}

/**
 * @brief 计算平面中点和线段的关系
 *
 * @param slp                   线段的起始点
 * @param elp                   线段的结束点
 * @param pt                    判断的点
 * @return  < 0 左, = 0 线上, > 0 右
 * @author wubo
 * @date 2024/05/27
 * @remark 1.2024/05/27 10:35 created by wubo
 */
template <typename T, typename T2>
int GetPonitLineRelation(const T &linePt1, const T &linePt2, const T2 &pt)
{
    double p1x = linePt1.x;
    double p1y = linePt1.y;
    double p2x = linePt2.x;
    double p2y = linePt2.y;
    double px = pt.x;
    double py = pt.y;

    double crossProduct = (p1y - py) * (p2x - px) - (p1x - px) * (p2y - py);

    // 如果叉积接近于零，则点在直线上
    double epsilon = 1e-6;
    if (std::abs(crossProduct) < epsilon) {
        return 0;    // 在线段上
    }

    return crossProduct;    // 返回有向距离
}

/**
 * @brief 判断平面中点是否在两点连成的线段上
 *
 * @param p                      // 待判断的点
 * @param a                      // 线段的起点
 * @param b                      // 线段的终点
 * @return true                  // 点在线段上
 * @return false                 // 点不在线段上
 * @author wubo
 * @date 2024/05/28
 * @remark 1.2024/05/28 15:35 created by wubo
 */
template <typename T, typename T2>
bool isPointOnSegment(const T &a, const T &b, const T2 &p)
{
    return GetPonitLineRelation(a, b, p) == 0 && std::min(a.x, b.x) <= p.x && p.x <= std::max(a.x, b.x)
           && std::min(a.y, b.y) <= p.y && p.y <= std::max(a.y, b.y);
}

/**
 * @brief 计算平面中点是否在多边形内
 *
 * @param bypts                       //边界点
 * @param int num                     //边界点数
 * @param pt                          //判断的点
 * @return 0 : 不在,  1：在边界上,  2：在边界内
 * @author wubo
 * @date 2024/05/28
 * @remark 1.2024/05/28 15:35 created by wubo
 */
template <typename T, typename T2>
int isPointInPolygon(const T *bypts, int num, const T2 &pt)
{
    bool isInside = false;

    // 使用射线法判断点是否在多边形内部
    for (int i = 0, j = num - 1; i < num; j = i++) {
        if (((bypts[i].y > pt.y) != (bypts[j].y > pt.y))
            && (pt.x < (bypts[j].x - bypts[i].x) * (pt.y - bypts[i].y) / (bypts[j].y - bypts[i].y) + bypts[i].x))
        {
            isInside = !isInside;
        }
    }

    if (isInside) {
        return 2;    // 点在多边形内部
    } else {
        // 检查点是否在多边形的边上
        for (int i = 0, j = num - 1; i < num; j = i++) {
            if (isPointOnSegment(bypts[i], bypts[j], pt)) {
                return 1;    // 点在多边形的边上
            }
        }
        return 0;    // 点在多边形外
    }
}

// 凸包算法辅助函数 比较两个点的大小
template <typename T>
bool compare(const T &p1, const T &p2)
{
    return p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y);
}

// 凸包算法辅助函数 判断三点的方向
template <typename T>
int threePtdir(const T &O, const T &A, const T &B)
{
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

/**
 * @brief 凸包算法
 *
 * @param pts                       //散点
 * @param int num                   //点数
 * @param std::vector<T>            //出参
 * @author wubo
 * @date 2024/05/28
 * @remark 1.2024/05/28 15:35 created by wubo
 */
template <typename T>
void ConvexHull(const T *pts, int num, std::vector<T> &out)
{
    if (num < 3) {
        return;    // 凸包的点数不能少于 3
    }

    std::vector<T> points(pts, pts + num);

    // 按照 x 坐标排序，如果 x 相同则按照 y 坐标排序
    std::sort(points.begin(), points.end(), compare<T>);

    std::vector<T> hull;

    // 下半部分
    for (const auto &p : points) {
        while (hull.size() >= 2 && threePtdir(hull[hull.size() - 2], hull.back(), p) <= 0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    // 上半部分
    size_t lowerHullSize = hull.size();
    for (int i = points.size() - 2; i >= 0; i--) {
        const auto &p = points[i];
        while (hull.size() > lowerHullSize && threePtdir(hull[hull.size() - 2], hull.back(), p) <= 0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    // 移除最后一个点，因为它和第一个点是同一个点
    hull.pop_back();

    // 输出结果
    out = hull;
}

/**
 * @brief BresenhamLine（布雷森汉姆线）算法
 *
 * @param x0                                起点 x 坐标
 * @param y0                                起点 y 坐标
 * @param x1                                终点 x 坐标
 * @param y1                                终点 y 坐标
 * return std::vector<Point2D<T>> &dots     包含两个int类型的结构 (POINT2D_INT)
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/28 15:35 created by wubo
 */
template <typename T>
void BresenhamLine(int x0, int y0, int x1, int y1, std::vector<Point2D<T>> &dots)
{
    int dx = x1 - x0;
    int dy = y1 - y0;

    if (std::abs(dx) >= std::abs(dy)) {
        // 如果dx的绝对值大于或等于dy的绝对值，按x轴步进
        if (x0 > x1) {
            std::swap(x0, x1);
            std::swap(y0, y1);
            dx = -dx;
            dy = -dy;
        }

        int error = -dx / 2;
        int y = y0;

        if (dy >= 0) {
            // 直线的倾斜角位于 [0, pi / 4]
            for (int x = x0; x <= x1; x++) {
                dots.push_back(Point2D<T>(x, y));
                error += dy;
                if (error >= 0) {
                    y++;
                    error -= dx;
                }
            }
        } else {
            // 直线的倾斜角位于 [-pi / 4, 0)
            for (int x = x0; x <= x1; x++) {
                dots.push_back(Point2D<T>(x, y));
                error += dy;
                if (error <= 0) {
                    y--;
                    error += dx;
                }
            }
        }
    } else {
        // 如果dy的绝对值大于dx的绝对值，按y轴步进
        if (y0 > y1) {
            std::swap(x0, x1);
            std::swap(y0, y1);
            dx = -dx;
            dy = -dy;
        }

        int error = -dy / 2;
        int x = x0;

        if (dx >= 0) {
            // 直线的倾斜角位于 (pi / 4, pi / 2]
            for (int y = y0; y <= y1; y++) {
                dots.push_back(Point2D<T>(x, y));
                error += dx;
                if (error >= 0) {
                    x++;
                    error -= dy;
                }
            }
        } else {
            // 直线的倾斜角位于 [-pi / 2, -pi / 4)
            for (int y = y0; y <= y1; y++) {
                dots.push_back(Point2D<T>(x, y));
                error += dx;
                if (error <= 0) {
                    x--;
                    error += dy;
                }
            }
        }
    }
}

/**
 * @brief 最近邻插值
 *
 * @param pt                      计算点
 * @param kdt                     kdt树
 * @param int dim                 kdt维度
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T, int dim>
double GetKDZ(const T &pt, void *kdtTree)
{
    POINTKDT findp(pt);
    KDT::KdTree<POINTKDT, dim> *kdt = (KDT::KdTree<POINTKDT, dim> *)kdtTree;
    return kdt->GetNearZ(findp);
}

/**
 * @brief IDW(反距离加权)
 *
 * @param pt                      计算点
 * @param points                  样本点
 * @param num                     样本点数
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T, typename T2>
double GetIDW(const T &pt, const T2 *pts, int num, int p = IDWC)
{
    double tmpDis = 0;
    double sumDis = 0;
    double r = 0;

    for (int i = 0; i < num; ++i) {
        tmpDis = Get2DDistance<T, T2>(pt, pts[i]);
        double tmpDisP = (pow(tmpDis, p) + EPSINON);

        sumDis += 1 / tmpDisP;
        r += pts[i].z / tmpDisP;
    }

    return r / sumDis;
}

/*******************************************************************************/
/*                          网格,索引计算                                      */
/*******************************************************************************/

/**
 * @brief 生成网格
 *
 * @param int dimX
 * @param int dimY
 * @param std::vector<MESHPOS> &out
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreateMeshingPos(int dimX, int dimY, std::vector<T> &out)
{
    T pos;
    for (int y = 0; y < dimY; y++) {
        for (int x = 0; x < dimX; x++) {
            pos.Pt[0] = glm::vec3(x, y, 0);
            pos.Pt[1] = glm::vec3(x + 1, y, 0);
            pos.Pt[2] = glm::vec3(x + 1, y + 1, 0);
            pos.Pt[3] = glm::vec3(x, y + 1, 0);
            out.push_back(pos);
        }
    }
}

/**
 * @brief 生成网格
 *
 * @param int dimX 网格的 X 方向尺寸
 * @param int dimY 网格的 Y 方向尺寸
 * @param float stepX X 方向的步长
 * @param float stepY Y 方向的步长
 * @param std::vector<MESHPOS> &out 生成的网格保存在这里
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreateMeshingPos(int dimX, int dimY, float stepX, float stepY, std::vector<T> &out)
{
    T pos;
    for (int y = 0; y < dimY; y++) {
        for (int x = 0; x < dimX; x++) {
            pos.Pt[0] = glm::vec3(x * stepX, y * stepY, 0);
            pos.Pt[1] = glm::vec3((x + 1) * stepX, y * stepY, 0);
            pos.Pt[2] = glm::vec3((x + 1) * stepX, (y + 1) * stepY, 0);
            pos.Pt[3] = glm::vec3(x * stepX, (y + 1) * stepY, 0);
            out.push_back(pos);
        }
    }
}

/**
 * @brief 生成矩形平面
 *
 * @param const T &minPos          //平面最小XY
 * @param const T &maxPos          //平面最大XY
 * @param const float &z           //平面Z值
 * @param std::vector<T> &outPos
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreatePlanePos(const T &minPos, const T &maxPos, const float &z, std::vector<T> &outPos)
{
    // 添加矩形的四个顶点
    outPos.push_back(T{minPos.x, minPos.y, z});
    outPos.push_back(T{maxPos.x, minPos.y, z});
    outPos.push_back(T{maxPos.x, maxPos.y, z});
    outPos.push_back(T{minPos.x, maxPos.y, z});
}

/**
 * @brief 生成矩形平面
 *
 * @param const T &minPos          //平面最小XY
 * @param const T &maxPos          //平面最大XY
 * @param const float &z           //平面Z值
 * @param MESHPOS &outPos
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreatePlanePos(const T &minPos, const T &maxPos, const float &z, MESHPOS &outPos)
{
    // 添加矩形的四个顶点
    outPos.Pt[0] = (T{minPos.x, minPos.y, z});
    outPos.Pt[1] = (T{maxPos.x, minPos.y, z});
    outPos.Pt[2] = (T{maxPos.x, maxPos.y, z});
    outPos.Pt[3] = (T{minPos.x, maxPos.y, z});
}

/**
 * @brief 生成矩形平面
 *
 * @param const T &center            //中心点
 * @param const float &width         //宽
 * @param const float &height        //高
 * @param std::vector<T> &outPos
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreatePlanePos(const T &center, const float &width, const float &height, std::vector<T> &outPos)
{
    // 清空输出向量
    outPos.clear();

    // 计算矩形的四个顶点
    T topLeft = center + T(-width / 2, height / 2, 0.0f);
    T topRight = center + T(width / 2, height / 2, 0.0f);
    T bottomLeft = center + T(-width / 2, -height / 2, 0.0f);
    T bottomRight = center + T(width / 2, -height / 2, 0.0f);

    // 第一个三角形
    outPos.push_back(topLeft);
    outPos.push_back(bottomLeft);
    outPos.push_back(topRight);

    // 第二个三角形
    outPos.push_back(topRight);
    outPos.push_back(bottomLeft);
    outPos.push_back(bottomRight);
}

/**
 * @brief 生成立方体
 *
 * @param const T &minPos          //平面最小XYZ
 * @param const T &maxPos          //平面最大XYZ
 * @param MESHPOS outPos[6]
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreateCubePos(const T &minPos, const T &maxPos, MESHPOS outPos[6])
{
    glm::vec3 vertices[8] = {
        {minPos.x, minPos.y, minPos.z},    // 0
        {maxPos.x, minPos.y, minPos.z},    // 1
        {maxPos.x, maxPos.y, minPos.z},    // 2
        {minPos.x, maxPos.y, minPos.z},    // 3
        {minPos.x, minPos.y, maxPos.z},    // 4
        {maxPos.x, minPos.y, maxPos.z},    // 5
        {maxPos.x, maxPos.y, maxPos.z},    // 6
        {minPos.x, maxPos.y, maxPos.z}     // 7
    };

    outPos[0] = (MESHPOS{vertices[0], vertices[1], vertices[2], vertices[3]});

    outPos[1] = (MESHPOS{vertices[4], vertices[5], vertices[6], vertices[7]});

    outPos[2] = (MESHPOS{vertices[0], vertices[3], vertices[7], vertices[4]});

    outPos[3] = (MESHPOS{vertices[1], vertices[2], vertices[6], vertices[5]});

    outPos[4] = (MESHPOS{vertices[3], vertices[2], vertices[6], vertices[7]});

    outPos[5] = (MESHPOS{vertices[0], vertices[1], vertices[5], vertices[4]});
}

/**
 * @brief 生成立方体
 *
 * @param const T &center          //中心点
 * @param const float &step        //网格步长
 * @param MESHPOS outPos[6]
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreateCubePos(const T &center, const float &step, MESHPOS outPos[6])
{
    glm::vec3 vertices[8] = {
        {center.x - step / 2, center.y - step / 2, center.z - step / 2},    // 0
        {center.x + step / 2, center.y - step / 2, center.z - step / 2},    // 1
        {center.x + step / 2, center.y + step / 2, center.z - step / 2},    // 2
        {center.x - step / 2, center.y + step / 2, center.z - step / 2},    // 3
        {center.x - step / 2, center.y - step / 2, center.z + step / 2},    // 4
        {center.x + step / 2, center.y - step / 2, center.z + step / 2},    // 5
        {center.x + step / 2, center.y + step / 2, center.z + step / 2},    // 6
        {center.x - step / 2, center.y + step / 2, center.z + step / 2}     // 7
    };

    // 前面 (z = center.z - step / 2)
    outPos[0] = MESHPOS{vertices[0], vertices[1], vertices[2], vertices[3]};
    // 后面 (z = center.z + step / 2)
    outPos[1] = MESHPOS{vertices[4], vertices[5], vertices[6], vertices[7]};
    // 左面 (x = center.x - step / 2)
    outPos[2] = MESHPOS{vertices[0], vertices[3], vertices[7], vertices[4]};
    // 右面 (x = center.x + step / 2)
    outPos[3] = MESHPOS{vertices[1], vertices[2], vertices[6], vertices[5]};
    // 上面 (y = center.y + step / 2)
    outPos[4] = MESHPOS{vertices[3], vertices[2], vertices[6], vertices[7]};
    // 下面 (y = center.y - step / 2)
    outPos[5] = MESHPOS{vertices[0], vertices[1], vertices[5], vertices[4]};
}

/**
 * @brief 矩形网格组织三角形
 *
 * @param const T &p0               //p0
 * @param const T &p1               //p1
 * @param const T &p2               //p2
 * @param const T &p3               //p3
 * @param std::vector<T> &outPos
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreatePlaneTriangles(const T &p0, const T &p1, const T &p2, const T &p3, std::vector<T> &outPos)
{
    outPos.push_back(p0);
    outPos.push_back(p1);
    outPos.push_back(p2);
    outPos.push_back(p2);
    outPos.push_back(p3);
    outPos.push_back(p0);
}

/**
 * @brief 矩形网格组织三角形
 *
 * @param const T &p0                //p0
 * @param const T &p1                //p1
 * @param const T &p2                //p2
 * @param const T &p3                //p3
 * @param std::vector<T> &outPos
 * @param std::vector<T> &outNor
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreatePlaneTriangles(
    const T &p0, const T &p1, const T &p2, const T &p3, std::vector<T> &outPos, std::vector<T> &outNor)
{
    glm::vec3 normal1, normal2;
    CalcuTriangleNormal(p0, p1, p2, normal1);
    CalcuTriangleNormal(p2, p3, p0, normal2);

    outPos.push_back(p0);
    outPos.push_back(p1);
    outPos.push_back(p2);
    std::fill_n(std::back_inserter(outNor), 3, normal1);

    outPos.push_back(p2);
    outPos.push_back(p3);
    outPos.push_back(p0);
    std::fill_n(std::back_inserter(outNor), 3, normal2);
}

/**
 * @brief 矩形网格组织三角形
 *
 * @param const MESHPOS &mesh
 * @param std::vector<T> &outPos
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreatePlaneTriangles(const MESHPOS &mesh, std::vector<T> &outPos)
{
    CreatePlaneTriangles(mesh.Pt[0], mesh.Pt[1], mesh.Pt[2], mesh.Pt[3], outPos);
}

/**
 * @brief 矩形网格组织三角形
 *
 * @param const MESHPOS &mesh
 * @param std::vector<T> &outPos
 * @param std::vector<T> &outNor
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreatePlaneTriangles(const MESHPOS &mesh, std::vector<T> &outPos, std::vector<T> &outNor)
{
    CreatePlaneTriangles(mesh.Pt[0], mesh.Pt[1], mesh.Pt[2], mesh.Pt[3], outPos, outNor);
}

/**
 * @brief 矩形网格组织网格线
 *
 * @param const T &p0                //p0
 * @param const T &p1                //p1
 * @param const T &p2                //p2
 * @param const T &p3                //p3
 * @param std::vector<T> &outPos
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreateMeshLine(const T &p0, const T &p1, const T &p2, const T &p3, std::vector<T> &outPos)
{
    outPos.push_back(p0);
    outPos.push_back(p1);
    outPos.push_back(p1);
    outPos.push_back(p2);
    outPos.push_back(p2);
    outPos.push_back(p3);
    outPos.push_back(p3);
    outPos.push_back(p0);
}

/**
 * @brief 矩形网格组织网格线
 *
 * @param const MESHPOS &mesh
 * @param std::vector<T> &outPos
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CreateMeshLine(const MESHPOS &mesh, std::vector<T> &outPos)
{
    CreateMeshLine(mesh.Pt[0], mesh.Pt[1], mesh.Pt[2], mesh.Pt[3], outPos);
}

/**
 * @brief 组织圆形坐标
 *
 * @param const T &center                       // 中心点
 * @param const float &radius                   // 半径
 * @param const std::vector<float> &angles      // 角度数组
 * @param std::vector<T> &outPos                // 输出坐标
 * @param std::vector<int> &outIdx              // 角度点数
 * @param float z = 0.0f                        // z值
 * @param int sliceNum = 100                    // 分段数
 */
template <typename T>
void CreateRoundPoint(T center,
                      float radius,
                      const std::vector<float> &angles,
                      std::vector<T> &outPos,
                      std::vector<int> &outIdx,
                      float z = 0.0f,
                      int sliceNum = 100)
{
    float cumulativeAngle = 0.0f;
    const int size = angles.size();

    // 预分配输出向量的容量
    outPos.reserve(size * (sliceNum + 1));

    for (const float &sliceAngle : angles) {
        float segmentAngle = sliceAngle / sliceNum;
        for (int j = 0; j <= sliceNum; j++) {
            T pos;
            float ang = cumulativeAngle + j * segmentAngle;
            float theta = glm::radians(ang);
            pos.x = center.x + radius * std::sin(theta);
            pos.y = center.y + radius * std::cos(theta);
            pos.z = z;
            outPos.push_back(pos);
        }

        cumulativeAngle += sliceAngle;
        outIdx.push_back(outPos.size() - 1);
    }
}

/**
 * @brief 组织圆形坐标
 *
 * @param const T &center                       // 中心点
 * @param const float &radius                   // 半径
 * @param const std::vector<float> &angles      // 角度数组
 * @param std::vector<T> &outPos                // 输出坐标
 * @param std::vector<int> &outIdx              // 角度点数
 * @param float z = 0.0f                        // z值
 * @param int sliceNum = 100                    // 分段数
 */
template <typename T>
void CreateRoundPoint(T center,
                      float radius,
                      float angle,
                      std::vector<T> &outPos,
                      std::vector<int> &outIdx,
                      float z = 0.0f,
                      int sliceNum = 100)
{
    int num = 360.0f / angle;
    std::vector<float> angles(num, angle);
    CreateRoundPoint(center, radius, angles, outPos, outIdx, z, sliceNum);
}

/**
 * @brief 计算平面中点是否在网格内
 *
 * @param mesh                        //网格
 * @param pt                          //判断的点
 * @return 0 : 不在,  1：在边界上,  2：在边界内
 * @author wubo
 * @date 2024/05/28
 * @remark 1.2024/05/28 15:35 created by wubo
 */
template <typename T>
int isPointInMesh(const MESHPOS &mesh, const T &pt)
{
    std::vector<glm::vec3> tmp(4);
    for (int j = 0; j < 4; j++) {
        tmp[j] = mesh.Pt[j];
    }

    return COMMON::isPointInPolygon(tmp.data(), tmp.size(), pt);
}

/**
 * @brief 计算网格中点
 *
 * @param const MESHPOS &pos                      网格点
 * @param out                                     出参
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void CalcuMeshCenter(const MESHPOS &pos, T &out)
{
    // 计算四个顶点的坐标总和
    glm::vec3 sum(0.0f);
    for (int i = 0; i < 4; ++i) {
        sum += pos.Pt[i];
    }

    // 计算平均值，即中点的坐标
    out = sum / 4.0f;
}

/**
 * @brief 寻找当前点的最近网格的XY索引
 *
 * @param const DIMS &dims                        网格维度
 * @param const std::vector<MESHPOS> &MeshPos     网格点
 * @param const T &pt                             寻找点
 * @param POINT2D_INT &outIdx                     出参索引
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
template <typename T>
void GetNearestMeshIndex(const DIMS &dims, const std::vector<MESHPOS> &meshs, const T &pt, POINT2D_INT &outIdx)
{
    // 初始化最近点的索引和最小距离
    int closestX = 0;
    int closestY = 0;
    float minDist = std::numeric_limits<float>::max();

    // 遍历所有网格点，找到离目标点最近的点
    for (int i = 0; i < meshs.size(); i++) {
        T center;
        const MESHPOS &meshPos = meshs[i];
        CalcuMeshCenter(meshPos, center);
        // 计算网格点和目标点之间的距离
        float dist = Get2DDistance<T, T>(center, pt);
        if (dist < minDist) {
            minDist = dist;
            GetIndexXY(i, dims.x, closestX, closestY);
        }
    }

    // 更新出参
    outIdx.x = closestX;
    outIdx.y = closestY;
}

/**
 * @brief x y 计算索引
 *
 * @param int x
 * @param int y
 * @param int dimx
 * @return idx
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
int GetXYIndex(int x, int y, int dimx);

/**
 * @brief 索引计算x y
 *
 * @param int idx
 * @param int dimx
 * @param int &x
 * @param int &y
 * @return idx
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
void GetIndexXY(int idx, int dimx, int &x, int &y);

/**
 * @brief 索引计算
 *
 * PS:判断当前方向合法下个索引
 *
 * @param const DIMS &dim
 * @param const GAMEDIR &dir
 * @param int &idx
 * @return bool  合法性,true不合法
 * @author wubo
 * @date 2024/05/29
 * @remark 1.2024/05/29 11:35 created by wubo
 */
bool GetMeshDirNextIdx(const DIMS &dim, const GAMEDIR &dir, int &idx);

}    // namespace COMMON

#endif    // !GRAPHICS_H_
