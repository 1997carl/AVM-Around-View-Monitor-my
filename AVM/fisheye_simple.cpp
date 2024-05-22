#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <sys/time.h>
//#include "/home/ub/tb_ros_ws/src/share_lib/shared_lib.h"//可以删除该行代码，测试使用的自定义库
using namespace std;
using namespace cv;
namespace fs = boost::filesystem;
#define SAVE_VIDEO_AVM 1
#define SAVE_VIDEO_TRANS 1
#define SAVE_IMG_TRANS 0
#define SAVE_IMG_UNDISTORT 0
#define SAVE_IMG_DISTORT 0

#define SHOW_IMG_TRANS 0
#define SHOW_IMG_UNDISTORT 0
#define SHOW_IMG_DISTORT 0

///////////////////////////////////////所需要全局的变量//////////////////////////////////////////
// 在全局作用域外定义并初始化两个坐标点对象
// 定义表示坐标点的结构体


struct Point
{
    int x;
    int y;
};
// 像素坐标系中的关键点，当前，一个像素代表0.5cm
::Point ori_point = {0, 0};
::Point point_left_up = {400, 400};
::Point point_right_up = {600, 400};
::Point point_left_down = {400, 800};
::Point point_right_down = {600, 800};
::Point AVM_point = {1000, 1200};
//////////////////////设置对应的图片尺寸
int AVM_WIDTH = AVM_point.x;
int AVM_HIGHT = AVM_point.y;
// 摄像机捕获的原始图像大小
int original_width = 1920;
int original_height = 1080;
// 去畸变后的图像大小（默认四个大小相同）
int undistorted_width = 1920;
int undistorted_height = 1080;
// 透视变换后的图像大小
// 前
int trans_width_f = AVM_point.x;
int trans_height_f = point_left_up.y;
// 左
int trans_width_l = AVM_point.y;
int trans_height_l = point_left_up.x;
// 右
int trans_width_r = AVM_point.y;
int trans_height_r = AVM_point.x - point_right_up.x;

// 后
int trans_width_b = AVM_point.x;
int trans_height_b = AVM_point.y - point_left_down.y;
///////////////直接显示的图片区域，图像大小
// 前非重叠区
int part_front_width = trans_width_f - trans_height_l - trans_height_r;
int part_front_hight = trans_height_f;

// 左非重叠区
int part_left_width = trans_width_l - trans_height_f - trans_height_b;
int part_left_hight = trans_height_l;

// 右非重叠区
int part_right_width = trans_width_r - trans_height_f - trans_height_b;
int part_right_hight = trans_height_r;

// 后非重叠区
int part_back_width = trans_width_b - trans_height_l - trans_height_r;
int part_back_hight = trans_height_b;

///////////////四个角落重叠区域的大小

// 左上重叠的宽高
int left_up_width = trans_height_l;
int left_up_height = trans_height_f;
// 左下重叠区域的宽高
int left_down_width = trans_height_l;
int left_down_height = trans_height_b;
// 右上重叠区域的宽高
int right_up_width = trans_height_r;
int right_up_height = trans_height_f;
// 右下重叠区域的宽高
int right_down_width = trans_height_r;
;
int right_down_height = trans_height_b;

// 设置每个图片的尺寸
int distort_width = 640;
int distort_hight = 480;

int undistort_width = 640;
int undistort_hight = 480;

// 读取图片1和背景图片
cv::Mat AVM_PIC; // 最后要得到的图像
cv::Mat distorted_f;
cv::Mat distorted_l;
cv::Mat distorted_r;
cv::Mat distorted_b; // 用来接受畸变图像

cv::Mat undistorted_f;
cv::Mat undistorted_l;
cv::Mat undistorted_r;
cv::Mat undistorted_b;// 用来接受畸变图像
// change step3       // 四个投影图片的大小
// cv::Mat transformed_image_f(trans_height_f, trans_width_f, CV_8UC3); // 先高后宽
// cv::Mat transformed_image_l(trans_height_l, trans_width_l, CV_8UC3);
// cv::Mat transformed_image_r(trans_height_r, trans_width_r, CV_8UC3);
// cv::Mat transformed_image_b(trans_height_b, trans_width_b, CV_8UC3);

cv::Mat transformed_image_f(trans_height_f, trans_width_f, CV_8UC3); // 先高后宽
cv::Mat transformed_image_l(trans_height_l, trans_width_l, CV_8UC3);
cv::Mat transformed_image_r(trans_height_r, trans_width_r, CV_8UC3);
cv::Mat transformed_image_b(trans_height_b, trans_width_b, CV_8UC3);

// 将背景图片转为带透明通道的图片
cv::Mat background_original; // 读取图片车辆模型的图片/home/ub/tb_ros_ws/src/tb_car/src/pic/car.png
cv::Mat background;          // 建立一个4通道的图片
cv::Mat background_copy;     // 复制的一个图片，用来调试位置
cv::Mat background_copy2;    // 复制的一个图片，用来调试位置

cv::Mat left_up(left_up_height, left_up_width, CV_8UC3);
cv::Mat left_down(left_down_height, left_down_width, CV_8UC3);
cv::Mat right_up(right_up_height, right_up_width, CV_8UC3);
cv::Mat right_down(right_down_height, right_down_width, CV_8UC3); // 四个重叠区域的图片

// 定义源图像和目标区域
cv::Rect left_up_in_AVM_rect(0, 0, left_up.cols, left_up.rows);                                            // 目标区域                全景左上角待粘贴的的位置
cv::Rect right_up_in_AVM_rect(point_right_up.x, 0, right_up.cols, right_up.rows);                          // 目标区域           全景右上角待粘贴的的位置
cv::Rect left_down_in_AVM_rect(0, point_left_down.y, left_down.cols, left_down.rows);                      // 目标区域        全景左下角待粘贴的的位置
cv::Rect right_down_in_AVM_rect(point_right_down.x, point_right_down.y, right_down.cols, right_down.rows); // 目标区域   全景右下角待粘贴的的位置

/////前视图定义的3个区域，直接显示区域，左上角重叠区域，右上角重叠区域
cv::Rect f_show_backgrand_f(left_up.cols, 0, part_front_width, part_front_hight);         // 源图像区域，        xy宽高  前视图直接显示在全景的区域
cv::Rect f_show_backgrand_backgrand(left_up.cols, 0, part_front_width, part_front_hight); // 目标区域             全景中前视图粘贴的位置
cv::Rect left_up_f_roi(0, 0, left_up.cols, left_up.rows);                                 // 左上方的重叠区域                      前视图中左上角重叠区域，
cv::Rect right_up_f_roi(point_right_up.x, 0, right_up.cols, right_up.rows);               // 左上方的重叠区域
// 定义源图像和目标区域
cv::Rect l_show_backgrand_l(0, left_up.rows, part_left_width, part_left_hight);         // 源图像区域，xy宽高
cv::Rect l_show_backgrand_backgrand(0, left_up.rows, part_left_width, part_left_hight); // 目标区域
cv::Rect left_up_l_roi(0, 0, left_up.cols, left_up.rows);                               // 左上方的重叠区域  特殊与left_up_f_roi一致
cv::Rect left_down_l_roi(0, point_right_down.y, right_down.cols, right_down.rows);      // 左侧图片中左下角重叠区域
// 右侧视图拆解的区域
cv::Rect r_show_backgrand_r(0, right_up.rows, part_right_width, part_right_hight);                        // 源图像区域，xy宽高
cv::Rect r_show_backgrand_backgrand(point_right_up.x, right_up.rows, part_right_width, part_right_hight); // 目标区域
cv::Rect right_up_r_roi(0, 0, right_up.cols, right_up.rows);                                              // 左上方的重叠区域  特殊与left_up_f_roi一致
cv::Rect right_down_r_roi(0, point_right_down.y, right_down.cols, right_down.rows);                       // 左侧图片中左下角重叠区域
// 后侧视图拆解的区域
cv::Rect b_show_backgrand_b(left_down.cols, 0, part_back_width, part_back_hight);           // 源图像区域，        xy宽高  前视图直接显示在全景的区域
cv::Rect b_show_backgrand_backgrand(point_left_down.x, point_left_down.y, part_back_width, part_back_hight); // 目标区域             全景中前视图粘贴的位置
cv::Rect left_down_b_roi(0, 0, left_down.cols, left_down.rows);                             // 左上方的重叠区域                      前视图中左上角重叠区域，
cv::Rect right_down_b_roi(point_right_up.x, 0, right_down.cols, right_down.rows);           // 左上方的重叠区域

// 权重图片
cv::Mat weight_left_up = cv::imread("/home/ub/tb_ros_ws/src/tb_car/src/avm_source/1000x1200/weights_left_up.png", cv::IMREAD_GRAYSCALE);
cv::Mat weight_right_up = cv::imread("/home/ub/tb_ros_ws/src/tb_car/src/avm_source/1000x1200/weights_right_up.png", cv::IMREAD_GRAYSCALE);
cv::Mat weight_left_down = cv::imread("/home/ub/tb_ros_ws/src/tb_car/src/avm_source/1000x1200/weights_left_down.png", cv::IMREAD_GRAYSCALE);
cv::Mat weight_right_down = cv::imread("/home/ub/tb_ros_ws/src/tb_car/src/avm_source/1000x1200/weights_right_down.png", cv::IMREAD_GRAYSCALE);
/////////////////////////

////////////////////////////////
cv::Mat K_F = (cv::Mat_<double>(3, 3) << 645.7346345499831,0.0,917.2230775240195, 0.0,646.9252270626394,533.7215531937753, 0.0, 0.0, 1.0);          //  鱼眼相机参数
cv::Mat K2_F = (cv::Mat_<double>(3, 3) << 645.7346345499831/3,0.0,917.2230775240195, 0.0,646.9252270626394/3,533.7215531937753, 0.0, 0.0, 1.0); // 请替换为实际的相机参数
cv::Mat D_F = (cv::Mat_<double>(1, 4) << -0.17760307717029078  ,0.0011259455950739764, 0.010727790826048831, -0.0033875618996366897);                // 请替换为实际的畸变系数
// 右
cv::Mat K_R = (cv::Mat_<double>(3, 3) << 635.6167303192635, 0.0, 950.3088864658412, 0.0, 637.6661433236408, 534.0449235530549, 0.0, 0.0, 1.0);          //  鱼眼相机参数
cv::Mat K2_R = (cv::Mat_<double>(3, 3) << 635.6167303192635/3, 0.0, 950.3088864658412, 0.0, 637.6661433236408/3, 534.0449235530549, 0.0, 0.0, 1.0); // 请替换为实际的相机参数
cv::Mat D_R = (cv::Mat_<double>(1, 4) << -0.19294334592278634,  0.06128854178639569, -0.03983744052720816, 0.011122763484171795);                       // 请替换为实际的畸变系数
// 左
cv::Mat K_L = (cv::Mat_<double>(3, 3) << 633.5831523766649, 0.0, 991.7038006675581, 0.0, 631.7411402387959, 581.8241754990946,0.0, 0.0, 1.0);          //  鱼眼相机参数
cv::Mat K2_L = (cv::Mat_<double>(3, 3) << 633.5831523766649/3, 0.0, 991.7038006675581, 0.0, 631.7411402387959/3, 581.8241754990946,0.0, 0.0, 1.0); // 请替换为实际的相机参数
cv::Mat D_L = (cv::Mat_<double>(1, 4) <<-0.1465683994291964, -0.043957461915232206, 0.04382011662428436, -0.012099863070512938);                      // 请替换为实际的畸变系数
// 后  焦距系数除以3
cv::Mat K_B = (cv::Mat_<double>(3, 3) << 688.392106103004,0.0,943.0078033926793, 0.0,688.1202848523649,537.4748437066706, 0.0, 0.0, 1.0);          //  鱼眼相机参数
cv::Mat K2_B = (cv::Mat_<double>(3, 3) << 688.392106103004/3,0.0,943.0078033926793, 0.0,688.1202848523649/3,537.4748437066706, 0.0, 0.0, 1.0); // 请替换为实际的相机参数
cv::Mat D_B = (cv::Mat_<double>(1, 4) << -0.2426671545341815,0.02159327781964237,0.016681177444146073,-0.007007143640462534);   

// 请替换为实际的畸变系数
// 初始化去畸变映射
cv::Mat map1_f, map2_f;
cv::Mat map1_l, map2_l;
cv::Mat map1_r, map2_r;
cv::Mat map1_b, map2_b;

// 透视变换矩阵
// 案例
// change step2
// #F
cv::Mat perspective_matrix_f = (cv::Mat_<double>(3, 3) << -1.0441870069093062, -1.5952205561407087, 1372.9912283839442 ,-0.13479091742260035, -2.3616985809378512, 1112.9318253275078, -0.00014477391258139628,-0.0032016547968017678, 1.0);
// #L      python格式矩阵
cv::Mat perspective_matrix_l = (cv::Mat_<double>(3, 3) << -1.3895661878052543, -2.7554409473998565, 2133.614556487745,-0.01763572573135652, -3.4222020132180213, 1578.232675369373, 0.00034508837635448134,-0.0046683924572289304, 1.0);
// #R
cv::Mat perspective_matrix_r = (cv::Mat_<double>(3, 3) << -0.7783481541637615, -1.6434867975667136, 1288.5692480425691,-0.08777326182120661, -1.73215641657984, 911.7286260752442, -1.4409725651990023e-05,-0.002820179451770405, 1.0);
// #B       python格式矩阵
cv::Mat perspective_matrix_b = (cv::Mat_<double>(3, 3) << -1.7551482062526151, -3.070762201119748, 2385.9890659781668,0.22994480513680868, -4.213220963628262, 1473.3803053204692, 0.00034218055994938426,-0.005955189784404123, 1.0);

///////4880P
///////4880P
// // #F
// cv::Mat perspective_matrix_f = (cv::Mat_<double>(3, 3) << -0.6993140030497873, -2.1419840521595224, 580.4670272118154, 0.020500162170505577, -2.35489772359856, 498.21483059912725, 0.00018094288016637925, -0.006025481353540986, 1.0);
// // #R
// cv::Mat perspective_matrix_r = (cv::Mat_<double>(3, 3) << -1.3809047567068626, -2.4571989667771343, 886.1993808039398, 0.005296395394128991, -2.594387253425836, 643.5296420170915, 2.7300991420999895e-05, -0.005483133965704422, 1.0);
// // #L      python格式矩阵
// cv::Mat perspective_matrix_l = (cv::Mat_<double>(3, 3) << -1.4441451469440751, -2.7141822200512773, 929.9865362756003, -0.06360944857247058, -2.486472237517548, 636.358691698366, -2.0684544845226194e-05, -0.005564437201170924, 1.0);
// // #B       python格式矩阵
// cv::Mat perspective_matrix_b = (cv::Mat_<double>(3, 3) << -0.44732798894830456, -1.5887447623117479, 520.0765144293722, -0.015540038114960564, -1.5656263934383514, 425.5260688640879, 6.0780770125205275e-06, -0.004210345450437996, 1.0);

struct combined_map
{
    cv::Mat mapx;
    cv::Mat mapy;
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 将前视摄像头进行透视变换，每个摄像头去畸变矩阵参数相同，透视变换矩阵各不相同。
int num_image = 0;
//////////////// 得到去畸变的函数
cv::Mat get_undistorted_image(cv::Mat &f_distorted_image, const cv::Mat &f_K, const cv::Mat &f_D, const cv::Mat &f_K2, cv::Mat &map1, cv::Mat &map2) // 需要传入图片，大小参数，透视变换矩阵
{
    cv::Mat undistorted_image;

    cv::remap(f_distorted_image, undistorted_image, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return undistorted_image; // 畸变图像的参数都相同
}

/////////////透视变换的函数
// 输入去畸变的函数，透视变换矩阵，透视变换后的图像大小

cv::Mat transform_undistorted_image(const cv::Mat &f_undistorted_image, int f_new_width, int f_new_height, const cv::Mat &f_perspective_matrix) // 需要传入图片，大小参数，透视变换矩阵
{
    cv::Mat transformed_image(f_new_height, f_new_width, CV_8UC3);
    cv::warpPerspective(f_undistorted_image, transformed_image, f_perspective_matrix, transformed_image.size());
    // cout << "trans img " << endl;
    // 获取图像大小
    int hight = transformed_image.rows;
    int width = transformed_image.cols;
    int channels = transformed_image.channels();
    // cout<<"\thight:"<<hight<<"\twidth:"<<width<<"\tchannels:"<<channels<<endl;
    return transformed_image;
}

//传入去畸变的map值，投影矩阵

combined_map getcombinemap( const cv::Mat &f_mapx_undistort, const cv::Mat &f_mapy_undistort, const cv::Mat &f_matrix,const int &trans_width,const int &trans_height)
{
//////////////////根据投影矩阵得到map值
    cv::Mat inverseTransMatrix;
    cv::invert(f_matrix, inverseTransMatrix);
    cv::Mat map_x, map_y, srcTM;
    cv::Mat map1_tran, map2_tran;
    // Generate the warp matrix
    srcTM = inverseTransMatrix.clone(); // If WARP_INVERSE, set srcTM to transformationMatrix
    cv::Size imgSize(trans_width, trans_height);
    map_x.create(imgSize, CV_32FC1);
    map_y.create(imgSize, CV_32FC1);

    double M11, M12, M13, M21, M22, M23, M31, M32, M33;
    M11 = srcTM.at<double>(0,0);
    M12 = srcTM.at<double>(0,1);
    M13 = srcTM.at<double>(0,2);
    M21 = srcTM.at<double>(1,0);
    M22 = srcTM.at<double>(1,1);
    M23 = srcTM.at<double>(1,2);
    M31 = srcTM.at<double>(2,0);
    M32 = srcTM.at<double>(2,1);
    M33 = srcTM.at<double>(2,2);

    for (int y = 0; y < imgSize.height; y++) {
        double fy = (double)y;
        for (int x = 0; x < imgSize.width; x++) {
            double fx = (double)x;
            double w = ((M31 * fx) + (M32 * fy) + M33);
            w = w != 0.0f ? 1.f / w : 0.0f;
            float new_x = (float)((M11 * fx) + (M12 * fy) + M13) * w;
            float new_y = (float)((M21 * fx) + (M22 * fy) + M23) * w;
            map_x.at<float>(y,x) = new_x;
            map_y.at<float>(y,x) = new_y;
    }
    }

    // fixed-point representation 转为定点计算效率更高（但实际测试效果不明显） 
    cv::Mat transformation_x, transformation_y;
    transformation_x.create(imgSize, CV_16SC2);
    transformation_y.create(imgSize, CV_16UC1);
    cv::convertMaps(map_x, map_y, map1_tran, map2_tran, false);
    combined_map combione_map;
    cv::remap(f_mapx_undistort, combione_map.mapx, map_x, map_y, cv::INTER_LINEAR);
    cv::remap(f_mapy_undistort, combione_map.mapy, map_x, map_y, cv::INTER_LINEAR);
    return combione_map;
}

cv::Mat mymerge(const cv::Mat &Image1_float, const cv::Mat &Image2_float, const cv::Mat &floatImage)
{
    cv::Mat imc;
    cv::Mat imA;
    cv::Mat imB;
    floatImage.convertTo(imc, CV_32F, 1.0 / 255.0);     // 权重图片的融合
    Image1_float.convertTo(imA, CV_32FC3, 1.0 / 255.0); // 图片0-1的原图片
    Image2_float.convertTo(imB, CV_32FC3, 1.0 / 255.0); // 图片是0-1的原图片
    // 分离通道
    std::vector<cv::Mat> channels1;
    cv::split(imA, channels1);
    std::vector<cv::Mat> channels2;
    cv::split(imB, channels2);
    std::vector<cv::Mat> result_channels;
    for (int i = 0; i < 3; i++)
    {
        cv::Mat channel_result = channels1[i].mul(1.0 - imc) + channels2[i].mul(imc);
        result_channels.push_back(channel_result);
    }
    // 合并通道
    cv::Mat result;
    cv::merge(result_channels, result);
    result.convertTo(result, CV_8U, 255);
    return result;
}

////////////全景图像拼接
// 输入五张图像，
// 返回 最后拼接成一幅全景环视图。 顺序：前左右后，中间车模

cv::Mat get_avm(cv::Mat &f_front, cv::Mat &f_left, cv::Mat &f_right, cv::Mat &f_back, cv::Mat &f_background) // 传入四幅图篇得到一副全景图片  //p_background为4通道图片
{
    //////////////////////////////////摄像头图片的透明转换//////////////////////////////////////
    //////////////////////裁剪拼接到背景
    //     cv::Rect f_show_backgrand_f(left_up.cols, 0, part_front_width,part_front_hight); // 源图像区域，        xy宽高  前视图直接显示在全景的区域
    // cv::Rect f_show_backgrand_backgrand (left_up.cols, 0, part_front_width, part_front_hight); // 目标区域             全景中前视图粘贴的位置

    // 将源图像的指定区域复制到目标区域
    cout << " start copy"<< "left_up.cols:" << left_up.cols << ",part_front_width:" << part_front_width << ",part_front_hight:" << part_front_hight << endl;
    cout << " start copy"<< "trans_width_f:" << trans_width_f << ",trans_height_l:" << trans_height_l << ",trans_height_r:" << trans_height_r << endl;
    f_front(f_show_backgrand_f).copyTo(f_background(f_show_backgrand_backgrand));
    cout << "f_front copy to f_background" << endl;
    f_left(l_show_backgrand_l).copyTo(f_background(l_show_backgrand_backgrand));
    f_right(r_show_backgrand_r).copyTo(f_background(r_show_backgrand_backgrand));
    f_back(b_show_backgrand_b).copyTo(f_background(b_show_backgrand_backgrand));
    //cout << "  f_back(b_show_backgrand_b).copyTo(f_background(b_show_backgrand_backgrand));" << endl;
    // 定义感兴趣的矩形区域

    // 左上角待融合区域
    cv::Mat left_up_f = f_front(left_up_f_roi); // 引用感兴趣区域的图像数据
    cv::Mat left_up_l = f_left(left_up_l_roi);  // 引用感兴趣区域的图像数据
    // 右上角待融合区域
    cv::Mat right_up_f = f_front(right_up_f_roi); // 引用感兴趣区域的图像数据
    cv::Mat right_up_r = f_right(right_up_r_roi); // 引用感兴趣区域的图像数据
    // 左下角待融合区域
    cv::Mat left_down_b = f_back(left_down_b_roi); // 引用感兴趣区域的图像数据
    cv::Mat left_down_l = f_left(left_down_l_roi); // 引用感兴趣区域的图像数据
    // 右下角待融合区域
    cv::Mat right_down_b = f_back(right_down_b_roi);  // 引用感兴趣区域的图像数据
    cv::Mat right_down_r = f_right(right_down_r_roi); // 引用感兴趣区域的图像数据

    // cout << "   f_back(b_show_backgrand_b).copyTo(f_background(b_show_backgrand_backgrand));" << endl;
    left_up = mymerge(left_up_l, left_up_f, weight_left_up);
    right_up = mymerge(right_up_r, right_up_f, weight_right_up);

    left_down = mymerge(left_down_l, left_down_b, weight_left_down);
    right_down = mymerge(right_down_r, right_down_b, weight_right_down);
    cout << "merge avm" << endl;

    left_up.copyTo(f_background(left_up_in_AVM_rect));
    right_up.copyTo(f_background(right_up_in_AVM_rect));
    left_down.copyTo(f_background(left_down_in_AVM_rect));
    right_down.copyTo(f_background(right_down_in_AVM_rect));
    // cout << "finish joint" << endl;
    return f_background;
}

int main()
{
    cv::FileStorage fs("./tb_car/output/picframe_time.xml", cv::FileStorage::WRITE);
    std::string description = "世界时间。对应的帧数";
    fs << "Description" << description;
    fs << "speedAndTime" << "{";

    cout << "set size end" << endl;
    ////////////////////////////////////////////////////////创建保存的位置////////////////////////////////////
    // cv::VideoWriter writer_distort_front;
    // writer_distort_front.open("./tb_car/src/tb_avm/video/front_distort.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(original_width, original_width), true);
    // cv::VideoWriter writer_distort_right;
    // writer_distort_right.open("./tb_car/src/tb_avm/video/right_distort.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(original_width, original_width), true);
    // cv::VideoWriter writer_distort_left;
    // writer_distort_left.open("./tb_car/src/tb_avm/video/left_distort.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(original_width, original_width), true);
    // cv::VideoWriter writer_distort_back;
    // writer_distort_back.open("./tb_car/src/tb_avm/video/back_distort.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(original_width, original_width), true);

    // cv::VideoWriter writer_undistort_front;
    // writer_undistort_front.open("./tb_car/src/tb_avm/video/front_undistort.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(undistorted_width, undistorted_height), true);
    // cv::VideoWriter writer_undistort_right;
    // writer_undistort_right.open("./tb_car/src/tb_avm/video/right_undistort.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(undistorted_width, undistorted_height), true);
    // cv::VideoWriter writer_undistort_left;
    // writer_undistort_left.open("./tb_car/src/tb_avm/video/left_undistort.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(undistorted_width, undistorted_height), true);
    // cv::VideoWriter writer_undistort_back;
    // writer_undistort_back.open("./tb_car/src/tb_avm/video/back_undistort.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(undistorted_width, undistorted_height), true);

    cv::VideoWriter writer_trans_front;
    writer_trans_front.open("./tb_car/src/tb_avm/video/front_trans.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(trans_width_f, trans_height_f), true);
    cv::VideoWriter writer_trans_left;
    writer_trans_left.open("./tb_car/src/tb_avm/video/left_trans.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(trans_height_l, trans_width_l), true); 
    cv::VideoWriter writer_trans_right;
    writer_trans_right.open("./tb_car/src/tb_avm/video/right_trans.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(trans_height_r, trans_width_r), true);
    cv::VideoWriter writer_trans_back;
    writer_trans_back.open("./tb_car/src/tb_avm/video/back_trans.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(trans_width_b, trans_height_b), true);
    cv::VideoWriter write_AVM;
    write_AVM.open("./tb_car/output/AVM.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(720, 1080), true);
    cv::VideoWriter write_AVM_without_info;
    write_AVM_without_info.open("./tb_car/output/AVM_noinfor.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(720, 1080), true);

    ////////////////////创建图片保存位置///////////////////////
    std::string distortPath = "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/distort_img";
    fs::create_directory(distortPath);
    std::string undistortPath = "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img";
    fs::create_directory(undistortPath);
    std::string transPath = "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/trans_img/";
    fs::create_directories(transPath);
    //////////////////////////////////////////结束位置创建/////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////开始摄像头的读取/////////////////
    cv::VideoCapture camera0(2); //  前
    cv::VideoCapture camera1(3); //  左
    cv::VideoCapture camera2(0); // 右
    cv::VideoCapture camera3(1); // 后

    camera0.set(cv::CAP_PROP_FRAME_WIDTH, original_width);
    camera0.set(cv::CAP_PROP_FRAME_HEIGHT, original_height);

    camera1.set(cv::CAP_PROP_FRAME_WIDTH, original_width);
    camera1.set(cv::CAP_PROP_FRAME_HEIGHT, original_height);

    camera2.set(cv::CAP_PROP_FRAME_WIDTH, original_width);
    camera2.set(cv::CAP_PROP_FRAME_HEIGHT, original_height);

    camera3.set(cv::CAP_PROP_FRAME_WIDTH, original_width);
    camera3.set(cv::CAP_PROP_FRAME_HEIGHT, original_height);
    //////////////////////////////////////结束摄像头的读取//////////////////////////////////

    ///////////////////////////开始 转换背景图为4通道图片////////////////////////////////////
    std::cout << "start get car.png" << std::endl;
    background = imread("/home/ub/tb_ros_ws/src/tb_car/src/avm_source/1000x1200/black_ground1000x1200.png", cv::IMREAD_COLOR);
    cout << "success change to 4 Chanel" << endl;                // 现在，imageWithAlpha 就是带有透明通道的图像，并且其数据类型为 CV_8UC4
    cv::imwrite("./tb_car/src/tb_avm/show_car.png", background); // 保存的背景图
    background_copy = background.clone();                        // 复制的背景图1
    background_copy2 = background.clone();                       // 复制的背景图2

    //////////////////////////// 结束转换背景图////////////////////////////
    int num_image = 0; // 记录循环次数，用于保存对应图片
    cv::Size new_size(original_width, original_height);
    // 开始去畸变
    cv::fisheye::initUndistortRectifyMap(K_F, D_F, cv::Mat(), K2_F, new_size, CV_16SC2, map1_f, map2_f);
    cv::fisheye::initUndistortRectifyMap(K_L, D_L, cv::Mat(), K2_L, new_size, CV_16SC2, map1_l, map2_l);
    cv::fisheye::initUndistortRectifyMap(K_R, D_R, cv::Mat(), K2_R, new_size, CV_16SC2, map1_r, map2_r);
    cv::fisheye::initUndistortRectifyMap(K_B, D_B, cv::Mat(), K2_B, new_size, CV_16SC2, map1_b, map2_b);
    combined_map combine_f= getcombinemap( map1_f,map2_f ,perspective_matrix_f , trans_width_f,trans_height_f );
    combined_map combine_l= getcombinemap( map1_l,map2_l ,perspective_matrix_l , trans_width_l,trans_height_l );
    combined_map combine_r= getcombinemap( map1_r,map2_r ,perspective_matrix_r , trans_width_r,trans_height_r );
    combined_map combine_b= getcombinemap( map1_b,map2_b ,perspective_matrix_b , trans_width_b,trans_height_b );
        struct timeval test;
        gettimeofday(&test, NULL);
        long long testTime_pre = (long long)test.tv_sec * 1000000 + (long long)test.tv_usec;
        long long testTime_cur;
    cout << "finish initial start whlie 1000x1200" << endl; // 初始化完成，循环显示全景图像
    int frame_count=0;
    while (true)
    {
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
        camera0 >> distorted_f;
        camera1 >> distorted_l; // 接受畸变图
        camera2 >> distorted_r; // 接受畸变图
        camera3 >> distorted_b; //
        // 用棋盘图片验证矩阵是否正确
        // distorted_f = cv::imread("/home/ub/tb_ros_ws/src/tb_car/src/avm_source/distorted_f.jpg", -1);
        // distorted_l = cv::imread("/home/ub/tb_ros_ws/src/tb_car/src/avm_source/distorted_l.jpg", -1);
        // distorted_r = cv::imread("/home/ub/tb_ros_ws/src/tb_car/src/avm_source/distorted_r.jpg", -1);
        // distorted_b = cv::imread("/home/ub/tb_ros_ws/src/tb_car/src/avm_source/distorted_b.jpg", -1);

        // cout<<"cols:"<<  distorted_f.cols<<"rows:"<< distorted_f.rows<<endl;//显示摄像头的尺寸
        double start_time = static_cast<double>(cv::getTickCount());
        gettimeofday(&test, NULL);
        testTime_cur = (int)test.tv_sec * 1000000 + (int)test.tv_usec;

//////////////////////////////显示鱼眼原始图像////////////////////////////////
#if SHOW_IMG_DISTORT
        cv::imshow("distorted_f", distorted_f);
        cv::imshow("distorted_l", distorted_l);
        cv::imshow("distorted_r", distorted_r);
        cv::imshow("distorted_b", distorted_b);
#else
        // std::cout << "未显示相机画面" << std::endl;
#endif
#if SAVE_IMG_DISTORT
        if (num_image < 100)
        {
            std::stringstream ssf;
            ssf << "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/distort_img/distorted_f" << (num_image) << ".jpg"; /// home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img
            cv::imwrite(ssf.str(), distorted_f);

            std::stringstream ssl;
            ssl << "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/distort_img/distorted_l" << (num_image) << ".jpg"; /// home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img
            cv::imwrite(ssl.str(), distorted_l);

            std::stringstream ssr;
            ssr << "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/distort_img/distorted_r" << (num_image) << ".jpg"; /// home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img
            cv::imwrite(ssr.str(), distorted_r);

            std::stringstream ssb;
            ssb << "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/distort_img/distorted_b" << (num_image) << ".jpg"; /// home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img
            cv::imwrite(ssb.str(), distorted_b);
            // cout<<"保存图片distort"<<endl;
        }
#else
        // std::cout << "未保存原始图片" << std::endl;
#endif
        ////////////////////////////////////////开始去畸变/////////////////////////
        // undistorted_f = get_undistorted_image(distorted_f, K_F, D_F, K2_F,map1_f,map2_f); // 读取全景前视的图片

        // undistorted_l = get_undistorted_image(distorted_l, K_L, D_L, K2_L,map1_l,map2_l); // 读取全景前视的图片

        // undistorted_r = get_undistorted_image(distorted_r, K_R, D_R, K2_R,map1_r,map2_r); // 读取全景前视的图片

        // undistorted_b = get_undistorted_image(distorted_b, K_B, D_B, K2_B,map1_b,map2_b); // 读取全景前视的图片
        // cout << "得到去畸变的图片" << endl;
        // cv::imshow("undistorted_b",undistorted_b);
        // cv::waitKey();
#if SHOW_IMG_UNDISTORT
        cv::imshow("front2", undistorted_f);
        cv::imshow("left2", undistorted_l);
        cv::imshow("right2", undistorted_r);
        cv::imshow("back2", undistorted_b);
        cv::waitKey();
#else
        // std::cout << "未显示去畸变画面" << std::endl;
#endif

#if SAVE_IMG_UNDISTORT
        if (num_image < 100)
        {
            cout << "saving" << endl;
            std::stringstream ss0;
            std::stringstream ss1;
            std::stringstream ss2;
            std::stringstream ss3;

            ss0 << "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img/undistort_front" << (num_image) << ".jpg"; /// home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img
            cv::imwrite(ss0.str(), undistorted_f);
            // cout << "saving front trans:" << ss0.str() << endl;

            ss1 << "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img/undistort_left" << (num_image) << ".jpg"; /// home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img
            cv::imwrite(ss1.str(), undistorted_l);
            // cout << "saving left trans:" << ss1.str() << endl;

            ss2 << "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img/undistort_right" << (num_image) << ".jpg"; /// home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img
            cv::imwrite(ss2.str(), undistorted_r);
            // cout << "saving right trans:" << ss2.str() << endl;

            ss3 << "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img/undistort_back" << (num_image) << ".jpg"; /// home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img
            cv::imwrite(ss3.str(), undistorted_b);
            // cout << "saving back trans:" << ss3.str() << endl;
        }
        // cout<<"End undistorted"<<endl;
#else
        // std::cout << "未保存去畸变图片" << std::endl;
#endif

        // ////////////////////////////////////////开始透视变换//////////////////////////////////////////////////
        // transformed_image_f = transform_undistorted_image(undistorted_f, trans_width_f, trans_height_f, perspective_matrix_f); // 需要传入图片，大小参数，透视变换矩阵
        // transformed_image_l = transform_undistorted_image(undistorted_l, trans_width_l, trans_height_l, perspective_matrix_l); // 需要传入图片，大小参数，透视变换矩阵
        // cv::rotate(transformed_image_l, transformed_image_l, cv::ROTATE_90_COUNTERCLOCKWISE);
        // transformed_image_r = transform_undistorted_image(undistorted_r, trans_width_r, trans_height_r, perspective_matrix_r); // 需要传入图片，大小参数，透视变换矩阵
        // cv::rotate(transformed_image_r, transformed_image_r, cv::ROTATE_90_CLOCKWISE);
        // // cout<<"End trans right"<<endl;
        //cv::Mat transformed_image_b_STEP2 = transform_undistorted_image(undistorted_b, trans_width_b, trans_height_b, perspective_matrix_b); // 需要传入图片，大小参数，透视变换矩阵
        // cv::rotate(transformed_image_b, transformed_image_b, cv::ROTATE_180);
        // cv::imshow("transformed_image_b_STEP2",transformed_image_b_STEP2);



        cv::remap(distorted_f, transformed_image_f, combine_f.mapx, combine_f.mapy, cv::INTER_LINEAR);
        cv::remap(distorted_l, transformed_image_l, combine_l.mapx, combine_l.mapy, cv::INTER_LINEAR);
        cv::rotate(transformed_image_l, transformed_image_l, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::remap(distorted_r, transformed_image_r, combine_r.mapx, combine_r.mapy, cv::INTER_LINEAR);
        cv::rotate(transformed_image_r, transformed_image_r, cv::ROTATE_90_CLOCKWISE);
        cv::remap(distorted_b, transformed_image_b, combine_b.mapx, combine_b.mapy, cv::INTER_LINEAR);
        cv::rotate(transformed_image_b, transformed_image_b, cv::ROTATE_180);
        // cv::imshow("transformed_image_b",transformed_image_b);
        // cv::waitKey();
        cout << "得到透视变换后的图片" << endl;
        cout<<transformed_image_f.cols<< " "<<transformed_image_f.rows<<endl;
        cout<<transformed_image_l.cols<< " "<<transformed_image_l.rows<<endl;
        cout<<transformed_image_r.cols<< " "<<transformed_image_r.rows<<endl;
        cout<<transformed_image_b.cols<< " "<<transformed_image_b.rows<<endl;
        // cv::imshow("transformed_image_b",transformed_image_b);
        // cv::waitKey();
        //  cout<<"End trans back"<<endl;

#if SHOW_IMG_TRANS
        cv::imshow("front3", transformed_image_f);
        cv::imshow("left3", transformed_image_l);
        cv::imshow("right3", transformed_image_r);
        cv::imshow("back3", transformed_image_b);
                // cv::waitKey();
        // cout<<"显示 trans"<<endl;
#else
        // std::cout << "未显示投影变换" << std::endl;
#endif

        // 保存对应的图片

#if SAVE_IMG_TRANS
        if (num_image < 100)
        {
            cout << "saving" << endl;
            std::stringstream ss0;
            std::stringstream ss1;
            std::stringstream ss2;
            std::stringstream ss3;

            ss0 << "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/trans_img_1000x1200/test_trans_front" << (num_image) << ".jpg"; /// home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img
            cv::imwrite(ss0.str(), transformed_image_f);
            // cout << "saving trans_front:" << ss0.str() << endl;

            ss1 << "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/trans_img_1000x1200/test_trans_left" << (num_image) << ".jpg"; /// home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img
            cv::imwrite(ss1.str(), transformed_image_l);
            // cout << "saving trans_left:" << ss1.str() << endl;

            ss2 << "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/trans_img_1000x1200/test_trans_right" << (num_image) << ".jpg"; /// home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img
            cv::imwrite(ss2.str(), transformed_image_r);
            // cout << "saving trans_right:" << ss2.str() << endl;

            ss3 << "/home/ub/tb_ros_ws/src/tb_car/src/tb_avm/trans_img_1000x1200/test_trans_back" << (num_image) << ".jpg"; /// home/ub/tb_ros_ws/src/tb_car/src/tb_avm/undistort_img
            cv::imwrite(ss3.str(), transformed_image_b);
            // cout << "saving trans_back:" << ss3.str() << endl;
        }
#else
        // std::cout << "图像文件：未保存透视变换图片" << std::endl;
#endif

#if SAVE_VIDEO_TRANS
        writer_trans_front.write(transformed_image_f);
        cout<<"保存视频trans f"<<endl;
        writer_trans_left.write(transformed_image_l);
        cout<<"保存视频trans l"<<endl;
        writer_trans_right.write(transformed_image_r);
        cout<<"保存视频trans r"<<endl;
        writer_trans_back.write(transformed_image_b);
        cout<<"保存视频all trans"<<endl;
#else
        cout << "视频文件：未保存透视变换视频" << endl;
#endif
        cout << "AVM_PIC start" << endl;
        // cv::waitKey();
        AVM_PIC = get_avm(transformed_image_f, transformed_image_l, transformed_image_r, transformed_image_b, background);
        // cout << "AVM_PIC finish" << endl;
        // cv::imshow("AVM_PIC",AVM_PIC);
#if SAVE_VIDEO_AVM

        cv::Rect roi(140, 0, 720, 1080); // 定义感兴趣区域（ROI）
        cv::Mat croppedImage = AVM_PIC(roi).clone();//保存对应图片
        cv::imwrite("/home/ub/tb_ros_ws/src/APS_PSD/picture/test_image/tbimg/avm_share.jpg",croppedImage);
        write_AVM_without_info.write(croppedImage);
        // // 构建文件名
        // std::stringstream ss;
        // ss << "/home/ub/tb_ros_ws/src/APS_PSD/input/" << num_image <<".jpg";
        // std::string file_name = ss.str();
        // 将图像保存到文件中


#endif
        // cv::waitKey();
        //  // 创建一个缩放后的图像
        //  cv::Mat scaledImage;
        //  double scale = 1;
        //  cv::resize(AVM_PIC, scaledImage, cv::Size(), scale, scale);
        //  // 在窗口中显示缩放后的图像
        double end_time = static_cast<double>(cv::getTickCount());
        double delay = (end_time - start_time) / cv::getTickFrequency() * 1000;
        std::stringstream ss2,ss3;
        ss2 << "delay:" << delay << "ms"<<"_curtime:"<<testTime_cur;
        ss3 <<"count:"<<frame_count;
        double testTime_cur_double = static_cast<double>(testTime_cur);
        fs <<"point"<< "{"<<"Time_cur"<<testTime_cur_double <<"count" << frame_count << "}";
        // writeImage(croppedImage,testTime_cur_double);//写入数据到缓存




        std::string text = ss2.str();
        std::string text2 = ss3.str();
        cv::Point position(50, 50);   
        cv::Point positiontext2(50, 100);             // 文字的位置
        int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体类型
        double fontScale = 1.0;                  // 字体大小
        cv::Scalar color(255, 255, 255);         // 字体颜色
        int thickness = 2;                       // 字体线条粗细
        
        cv::putText(croppedImage, text, position, fontFace, fontScale, color, thickness);
        cv::putText(croppedImage, text2, positiontext2, fontFace, fontScale, color, thickness);
        cv::imshow("AVM", croppedImage);

        

#if SAVE_VIDEO_AVM
        write_AVM.write(croppedImage);
        ////////////////////////////////保存世界时间和对应第几帧画面
        frame_count=frame_count+1;



        // cout<<"保存视频avm"<<endl;

#else
        cout << "not SAVEING AVM" << endl;
#endif
        num_image++;
        cout << "完成一次循环" << endl;
    } // 结束循环
    fs << "}";  // 关闭序列
    fs.release();
    //////////////////////////////////////////////////////////////////////退出程序//////////////////////////////////
    /////////////////////////////释放保存视频内存//////////////////
    // writer_undistort_front.release();
    // writer_undistort_right.release();
    // writer_undistort_left.release();
    // writer_undistort_back.release();

    // writer_distort_front.release();
    // writer_distort_left.release();
    // writer_distort_right.release();
    // writer_distort_back.release();

    writer_trans_front.release();
    writer_trans_left.release();
    writer_trans_right.release();
    writer_trans_back.release();

    write_AVM.release();
    ////////////////////////////////////结束视频的保存//////////////
    cv::destroyAllWindows();
    cout << "release all  and finish cpp" << endl;
#if SAVE_VIDEO_AVM
    cout << "保存:视频avm" << endl;
#else
    cout << "not SAVEING AVM" << endl;
#endif

#if SAVE_VIDEO_TRANS
    cout << "保存:透视变换视频" << endl;
#else
    cout << "未保存:透视变换视频" << endl;
#endif

#if SAVE_IMG_TRANS
    cout << "保存：透视变换图片" << endl;
#else
    cout << "未保存：透视变换视频" << endl;
#endif

#if SAVE_IMG_UNDISTORT
    cout << "保存：去畸变的图片" << endl;
#else
    cout << "未保存：去畸变的图片" << endl;
#endif

#if SAVE_IMG_DISTORT
    cout << "保存：原始鱼眼图片" << endl;
#else
    cout << "未保存：原始鱼眼图片" << endl;
#endif

// #define SAVE_VIDEO_TRANS 0
// #define SAVE_IMG_TRANS 1
// #define SAVE_IMG_UNDISTORT 0
// #define SAVE_IMG_DISTORT 0
    return 0;
}