#include "Triangle.h"
#include "rasterizer.h"
#include <Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Matrix4f get_view_matrix(Vector3f eye_pos) {
    Matrix4f view = Matrix4f::Identity();

    Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0],
            0, 1, 0, -eye_pos[1],
            0, 0, 1, -eye_pos[2],
            0, 0, 0, 1;

    view = translate * view;

    return view;
}

Matrix4f get_model_matrix(float rotation_angle) {
    Matrix4f model = Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    float rad = rotation_angle * MY_PI / 180;
    model(0, 0) = cos(rad);
    model(1, 1) = cos(rad);
    model(0, 1) = -sin(rad);
    model(1, 0) = sin(rad);
    return model;
}

Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                               float zNear, float zFar) {
    // Students will implement this function

    Matrix4f projection = Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.

    // 将角度转换为弧度
    float rad_fov = eye_fov * MY_PI / 180.0f;

    // 计算顶部和右侧的裁剪边界
    float t = tan(rad_fov / 2) * zNear; // 计算近裁剪面顶部的高度
    float r = t * aspect_ratio; // 计算近裁剪面右侧的宽度

    // 填充投影矩阵
    projection(0, 0) = zNear / r; // 水平方向的缩放
    projection(1, 1) = zNear / t; // 垂直方向的缩放
    projection(2, 2) = -(zFar + zNear) / (zFar - zNear); // 远近裁剪面的映射
    projection(2, 3) = -(2 * zFar * zNear) / (zFar - zNear); // 透视缩放
    projection(3, 2) = -1.0f;
    projection(3, 3) = 0.0f;

    return projection;
}

int main(int argc, const char **argv) {
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        } else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Vector3f eye_pos = {0, 0, 5};

    std::vector<Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        } else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
