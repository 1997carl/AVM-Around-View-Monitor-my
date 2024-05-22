# 个人代码

#### 介绍
This is a vehicle surround view system based on fisheye cameras. The system generates a bird's-eye view of the vehicle's surroundings by processing raw images from four fisheye cameras. 这是一个基于鱼眼相机的车辆全景环视系统，旨在通过传入四个鱼眼相机的原始图像，生成车辆周围的鸟瞰图。 该系统通过四个安装在车辆前后左右的鱼眼相机获取环境图像，并对这些图像进行处理，生成一幅车辆周围的鸟瞰图。项目请看https://github.com/1997carl/AVM-Around-View-Monitor-my}

#### 软件架构
1. **鱼眼图像采集模块**
2. **鱼眼图像去畸变表建立**
3. **透视变换表建立**
4. **图像融合模块**
5. **全景环视图像生成**


#### 安装教程

1.  车辆全景环视系统的安装教程

   本文档将指导您如何使用CMake编译和运行车辆全景环视系统，而无需依赖ROS系统。

   #### 1. 环境准备

   确保您的系统已安装以下软件和工具：

   1. **操作系统**：Ubuntu 16.04或更高版本
   2. **CMake**：2.8.3或更高版本
   3. **C++编译器**：支持C++11标准的编译器（如GCC 4.8或更高版本）
   4. **OpenCV**：用于图像处理的OpenCV库

   ##### 1.1 安装OpenCV

   如果尚未安装OpenCV，可以使用以下命令安装：

   ```
   bash复制代码sudo apt update
   sudo apt install libopencv-dev
   ```

   #### 2. 下载项目代码

   将项目代码克隆到本地目录：

   ```
   bash复制代码git clone <your_project_repository_url> tb_car
   cd tb_car
   ```

   #### 3. 编译项目

   创建一个构建目录并使用CMake进行构建：

   ```
   bash复制代码mkdir build
   cd build
   cmake ..
   make
   ```

   #### 4. 运行项目

   编译完成后，可以通过以下命令运行可执行文件：

   ```
   bash复制代码./tb_car
   ./tb_car_get_speed
   ./video
   ./fisheye_simple
   ```

#### 使用说明

1. 环境需求

   要运行这个项目，你需要以下环境和依赖：

   1. **操作系统**：Ubuntu 16.04或更高版本（推荐使用ROS的LTS版本）
   2. **编译工具**：CMake 2.8.3或更高版本
   3. **C++编译器**：支持C++11标准的编译器（如GCC 4.8或更高版本）
   4. **OpenCV**：用于图像处理的OpenCV库

### 修改为自己的项目：

依次运行文件夹的文件，得到投影矩阵和像计内参。应用于全景环视系统
输入原始图像，输出去畸变图像，4个相机内参，投影图像，4个投影矩阵

文件结构如下：

├─output_mergepic_matrix
│  ├─AVM_PIC_LOG
│  ├─mergepic
│  └─remapmatrix
├─source
│  ├─input_for_text
│  └─merge
└─tools
    ├─get_matrix_map
    │  ├─get_K_D_matrix
    │  │  ├─back
    │  │  ├─findcheekboard
    │  │  ├─front
    │  │  ├─left
    │  │  ├─output
    │  │  └─right
    │  └─get_Trans_matrix
    │      ├─back
    │      ├─front
    │      ├─left
    │      ├─output
    │      └─right
    └─get_mergepic
        └─output

全景环视系统运行所需环境opencv3.5.0

1.运行demo。

编译并运行。根目录下的fisheye_simple.cpp，将调用fisheye_simple.cpp根目录下的source文件下的。背景图片，原始鱼眼图片，融合区域权值图片。系统将显示全景环视图片。



2.修改成自己项目所需要的图片。

由于鱼眼相机的不同，以及全景环视图片的分辨率大小的不同。需要进行，鱼眼图片，透视变换图片，全景环视图片大小的修改。此外，从鱼眼相机图片到透视变换图片，映射关系的查找表需要修改。

因此修改步骤如下：

2.1.得到相机内参矩阵

进入tools\get_matrix_map\get_K_D_matrix文件夹。运行1080p_get_K_D.py文件   f，b，l，r分别对应前后左右四个方向。（fblr代表四个方向，其余代码同理）

脚本文件，将遍历对应文件夹下的鱼眼相机原始图片，得到相机内参，并保存到output_mergepic_matrix\remapmatrix\中.如calibration_data_front_1080p.yaml。

2.2.得到透视变换的矩阵

进入\tools\get_matrix_map\get_Trans_matrix\文件夹。运行四个1000x1200_transfisheye_.py文件   

脚本文件，将读取点对点映射，并得到透视变换的映射关系表，并保存到output_mergepic_matrix\remapmatrix\中.如1080p_transfor_front_1000x1200.yaml。

2.3.建立融合权值表

进入\tools\get_mergepic\文件夹。运行四个getmask_xx.py文件，left-up对应左上角融合区域。   

脚本文件，根据设置的多边形融合区域，权值随像素位置渐变，得到融合权值图，并保存权值图到\output_mergepic_matrix\mergepic\中.如weights_left_down.png。

2.4.修改fisheye_simple.cpp中路径，和宏命令。保存全景环视图片到所需要的文件夹。默认保存到\output_mergepic_matrix\AVM_PIC_LOG。

源代码中，将相机内参矩阵,透视变换的矩阵，替换为计算得到的矩阵参数。

如cv::Mat K_F  cv::Mat K2_F   cv::Mat D_F，修改为所需要的相机内参值 

cv::Mat perspective_matrix_f 

cv::Mat perspective_matrix_l 

cv::Mat perspective_matrix_r 

cv::Mat perspective_matrix_b 

修改为所需要的值。代码见174行。

实际项目，请替换AVM_code\source\merge中的融合图片为\output_mergepic_matrix\mergepic\中得到的weights_left_down.png。

2.5不同项目分辨率不同，注意透视变换矩阵计算的脚本和融合图片计算脚本中像素大小的修改。

进行fisheye_simple.cpp中鱼眼图片，透视变换图片，全景环视图片大小的修改。



需要修改代码如下。

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

2.5重新编译并运行fisheye_simple.cpp。



#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
