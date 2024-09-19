// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.h"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions) {
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices) {
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols) {
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Vector3f &v3, float w = 1.0f) {
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f *_v) {
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Vector3f v0 = _v[0];
    Vector3f v1 = _v[1];
    Vector3f v2 = _v[2];

    Vector3f p(x, y, 1); // 需要判断的点

    bool allPos = true;
    bool allNeg = true;

    Vector3f edge0 = v1 - v0;
    Vector3f vec0 = p - v0;
    const auto cross0 = edge0[0] * vec0[1] - edge0[1] * vec0[0];
    if (cross0 >= 0.0f) {
        allNeg = false;
    } else {
        allPos = false;
    }

    Vector3f edge1 = v2 - v1;
    Vector3f vec1 = p - v1;
    const auto cross1 = edge1[0] * vec1[1] - edge1[1] * vec1[0];
    if (cross1 >= 0.0f) {
        allNeg = false;
    } else {
        allPos = false;
    }

    Vector3f edge2 = v0 - v2;
    Vector3f vec2 = p - v2;
    const auto cross2 = edge2[0] * vec2[1] - edge2[1] * vec2[0];
    if (cross2 >= 0.0f) {
        allNeg = false;
    } else {
        allPos = false;
    }

    return allNeg || allPos;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f *v) {
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (
                   v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x()
                   * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (
                   v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x()
                   * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (
                   v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x()
                   * v[0].y());
    return {c1, c2, c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type) {
    auto &buf = pos_buf[pos_buffer.pos_id];
    auto &ind = ind_buf[ind_buffer.ind_id];
    auto &col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Matrix4f mvp = projection * view * model;
    for (auto &i: ind) {
        Triangle t;
        Vector4f v[] = {
            mvp * to_vec4(buf[i[0]], 1.0f),
            mvp * to_vec4(buf[i[1]], 1.0f),
            mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto &vec: v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto &vert: v) {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i) {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle &t) {
    auto v = t.toVector4();

    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    // 1. 计算三角形的包围盒子
    auto min_x = std::floor(std::min({v[0].x(), v[1].x(), v[2].x()}));
    auto max_x = std::ceil(std::max({v[0].x(), v[1].x(), v[2].x()}));
    auto min_y = std::floor(std::min({v[0].y(), v[1].y(), v[2].y()}));
    auto max_y = std::ceil(std::max({v[0].y(), v[1].y(), v[2].y()}));

    // 超采样采样率，比如 2x2 的 SSAA
    int ssaa_factor = 2; // 可以设置为 2x2 或 4x4
    float step = 1.0f / ssaa_factor; // 每个子像素的步长

    // 存储每个子像素的颜色和 z 值，后续用于合并
    std::vector sub_pixel_colors(ssaa_factor * ssaa_factor, Vector3f(0, 0, 0));
    std::vector sub_pixel_depths(ssaa_factor * ssaa_factor, std::numeric_limits<float>::infinity());


    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.

    for (int x = static_cast<int>(min_x); x < static_cast<int>(max_x); x++) {
        for (int y = static_cast<int>(min_y); y < static_cast<int>(max_y); y++) {
            for (int i = 0; i < ssaa_factor; i++) {
                for (int j = 0; j < ssaa_factor; j++) {
                    auto sub_x = x + (i + 0.5f) * step;
                    auto sub_y = y + (j + 0.5f) * step;

                    if (insideTriangle(sub_x, sub_y, t.v)) {
                        auto [alpha, beta, gamma] = computeBarycentric2D(sub_x, sub_y, t.v); // 计算重心坐标

                        float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                        float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].
                                               z() / v[
                                                   2].w();
                        z_interpolated *= w_reciprocal;

                        // 计算子像素的颜色
                        const auto color = t.getColor();
                        if (const int sub_index = i * ssaa_factor + j; z_interpolated < sub_pixel_depths[sub_index]) {
                            sub_pixel_colors[sub_index] = color;
                            sub_pixel_depths[sub_index] = z_interpolated;
                        }
                    }

                    // 合并子像素，计算像素的最终颜色

                    Vector3f final_color(0, 0, 0);
                    for (const auto &sub_color: sub_pixel_colors) {
                        final_color += sub_color;
                    }
                    final_color /= static_cast<float>(ssaa_factor * ssaa_factor);
                    float final_depth = 0;
                    for (const auto &sub_depth: sub_pixel_depths) {
                        final_depth += sub_depth;
                    }
                    final_depth /= static_cast<float>(ssaa_factor * ssaa_factor);

                    set_pixel(Vector3f(x, y, final_depth), final_color);
                }
            }


            // 不考虑SSAA的做法
            // if (insideTriangle(x + 0.5, y + 0.5, t.v)) {
            //     // If so, use the following code to get the interpolated z value.
            //     // 计算重心坐标
            //     auto [alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
            //     float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            //     float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[
            //                                2].w();
            //     z_interpolated *= w_reciprocal;
            //
            //     // 深度缓冲
            //     if (int index = get_index(x, y); z_interpolated < depth_buf[index]) {
            //         // 深度比缓冲中的深度更前的话更新深度缓冲
            //         depth_buf[index] = z_interpolated;
            //
            //         auto color = t.getColor();
            //         set_pixel(Vector3f(x, y, z_interpolated), color);
            //     }
            // }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f &m) {
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f &v) {
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f &p) {
    projection = p;
}

void rst::rasterizer::clear(Buffers buff) {
    if ((buff & Buffers::Color) == Buffers::Color) {
        std::fill(frame_buf.begin(), frame_buf.end(), Vector3f{0, 0, 0});
    }
    if ((buff & Buffers::Depth) == Buffers::Depth) {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h) {
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y) {
    return (height - 1 - y) * width + x;
}

void rst::rasterizer::set_pixel(const Vector3f &point, const Vector3f &color) {
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height - 1 - point.y()) * width + point.x();
    frame_buf[ind] = color;
}

// clang-format on
