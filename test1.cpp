#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void RGBToNV12(const Mat &rgbImage, vector<uint8_t> &nv12Data) {
    int width = rgbImage.cols;
    int height = rgbImage.rows;

    nv12Data.resize(width * height * 3 / 2); // Y + UV

    uint8_t *yPlane = nv12Data.data();
    uint8_t *uvPlane = nv12Data.data() + (width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Vec3b bgr = rgbImage.at<Vec3b>(y, x);
            uint8_t r = bgr[2];
            uint8_t g = bgr[1];
            uint8_t b = bgr[0];

            // 计算 Y 分量
            int Y = static_cast<int>(0.299 * r + 0.587 * g + 0.114 * b);
            Y = clamp(Y, 0, 255);
            yPlane[y * width + x] = static_cast<uint8_t>(Y);// Y

            // 计算 UV 分量
            if (x % 2 == 0 && y % 2 == 0) {
                int U = static_cast<int>(-0.14713 * r - 0.28886 * g + 0.436 * b) + 128;
                int V = static_cast<int>(0.615 * r - 0.51499 * g - 0.10001 * b) + 128;
                U = clamp(U, 0, 255);
                V = clamp(V, 0, 255);
                uvPlane[(y / 2) * (width / 2) + (x / 2) * 2] = static_cast<uint8_t>(U); // U
                uvPlane[(y / 2) * (width / 2) + (x / 2) * 2 + 1] = static_cast<uint8_t>(V); // V
            }
        }
    }
}

void NV12ToRGB(const vector<uint8_t> &nv12Data, Mat &rgbImage, int width, int height) {
    rgbImage.create(height, width, CV_8UC3);
    const uint8_t *yPlane = nv12Data.data();
    const uint8_t *uvPlane = nv12Data.data() + (width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int Y = yPlane[y * width + x];
            int U, V;

            if (x % 2 == 0 && y % 2 == 0) {
                U = uvPlane[(y / 2) * (width / 2) + (x / 2) * 2];
                V = uvPlane[(y / 2) * (width / 2) + (x / 2) * 2 + 1];
            } else if (x % 2 == 1 && y % 2 == 0) {
                U = uvPlane[(y / 2) * (width / 2) + (x / 2) * 2];
                V = uvPlane[(y / 2) * (width / 2) + (x / 2) * 2 + 1];
            } else if (x % 2 == 0 && y % 2 == 1) {
                U = uvPlane[((y - 1) / 2) * (width / 2) + (x / 2) * 2];
                V = uvPlane[((y - 1) / 2) * (width / 2) + (x / 2) * 2 + 1];
            } else {
                U = uvPlane[((y - 1) / 2) * (width / 2) + (x / 2) * 2];
                V = uvPlane[((y - 1) / 2) * (width / 2) + (x / 2) * 2 + 1];
            }

            // 计算 RGB 分量
            int r = Y + 1.402 * (V - 128);
            int g = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128);
            int b = Y + 1.772 * (U - 128);

            r = clamp(r, 0, 255);
            g = clamp(g, 0, 255);
            b = clamp(b, 0, 255);

            rgbImage.at<Vec3b>(y, x) = Vec3b(b, g, r);
        }
    }
}

int main() {
    // 读取图片
    Mat rgbImage = imread("/home/shijian/桌面/视觉测试/3.jpg");
    if (rgbImage.empty()) {
        cerr << "无法读取图片。" << endl;
        return -1;
    }

    // 转换 RGB 到 NV12
    vector<uint8_t> nv12Data;
    auto start = chrono::high_resolution_clock::now();
    RGBToNV12(rgbImage, nv12Data);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "RGB 转 NV12 用时: " << elapsed.count() << "秒" << endl;

    // 保存 NV12 文件
    ofstream nv12File("/home/shijian/图片/test1_nv12.yuv", ios::binary);
    nv12File.write(reinterpret_cast<const char *>(nv12Data.data()), nv12Data.size());
    nv12File.close();   

    // 转换 NV12 回 RGB
    Mat reconstructedImage;
    start = chrono::high_resolution_clock::now();
    NV12ToRGB(nv12Data, reconstructedImage, rgbImage.cols, rgbImage.rows);
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << "NV12 转 RGB 用时: " << elapsed.count() << "秒" << endl;

    // 保存重建后的图片
    imwrite("/home/shijian/图片/test1_reconstructed.jpg", reconstructedImage);

    return 0;
}
