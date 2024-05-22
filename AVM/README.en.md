# 个人代码

1. Personal Code

   #### Introduction

   This is a vehicle surround view system based on fisheye cameras. The system generates a bird's-eye view of the vehicle's surroundings by processing raw images from four fisheye cameras. This system acquires environmental images through four fisheye cameras installed at the front, rear, left, and right of the vehicle, processes these images, and generates a bird's-eye view of the vehicle's surroundings. For the project, please see [GitHub](https://github.com/1997carl/AVM-Around-View-Monitor-my).

   #### Software Architecture

   1. **Fisheye Image Acquisition Module**
   2. **Fisheye Image Distortion Correction Table Generation**
   3. **Perspective Transformation Table Generation**
   4. **Image Fusion Module**
   5. **Panoramic Surround View Image Generation**

   #### Installation Guide

   1. Vehicle Surround View System Installation Guide

      This document will guide you on how to compile and run the vehicle surround view system using CMake without relying on the ROS system.

      #### 1. Environment Preparation

      Ensure that your system has the following software and tools installed:

      1. **Operating System**: Ubuntu 16.04 or higher
      2. **CMake**: Version 2.8.3 or higher
      3. **C++ Compiler**: A compiler that supports the C++11 standard (e.g., GCC 4.8 or higher)
      4. **OpenCV**: The OpenCV library for image processing

      ##### 1.1 Installing OpenCV

      If OpenCV is not already installed, you can install it using the following commands:

      ```bash
      sudo apt update
      sudo apt install libopencv-dev

#### 2. Download Project Code

Clone the project code to a local directory:

```
bash复制代码git clone <your_project_repository_url> tb_car
cd tb_car
```

#### 3. Compile the Project

Create a build directory and use CMake to build the project:

```
bash复制代码mkdir build
cd build
cmake ..
make
```

#### 4. Run the Project

After compilation, you can run the executables with the following commands:

```
bash复制代码./tb_car
./tb_car_get_speed
./video
./fisheye_simple
```

#### Instructions

1. Environmental Requirements

   To run this project, you need the following environment and dependencies:

   1. **Operating System**: Ubuntu 16.04 or higher (preferably the LTS version of ROS)
   2. **Build Tools**: CMake 2.8.3 or higher
   3. **C++ Compiler**: A compiler that supports the C++11 standard (e.g., GCC 4.8 or higher)
   4. **OpenCV**: The OpenCV library for image processing

### Modifying for Your Own Project:

Run the files in sequence to get the projection matrix and camera parameters. Apply them to the surround view system to input raw images and output distortion-corrected images, camera parameters for four cameras, projected images, and four projection matrices.

The file structure is as follows:

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

The surround view system requires OpenCV 3.5.0.

1. Run the demo.

Compile and run. The fisheye_simple.cpp in the root directory will call the fisheye_simple.cpp file under the source folder in the root directory. The background images, original fisheye images, and fusion region weight images are used. The system will display the surround view images.

1. Modify the images needed for your project.

Due to differences in fisheye cameras and surround view image resolution, modifications are needed for fisheye images, perspective transformation images, and surround view image sizes. Additionally, the lookup table for mapping from fisheye images to perspective transformation images needs to be modified.

The modification steps are as follows:

2.1. Obtain the camera intrinsic matrix

Go to the tools\get_matrix_map\get_K_D_matrix folder. Run the 1080p_get_K_D.py file. f, b, l, r correspond to the front, rear, left, and right directions respectively (fblr represents four directions, the rest of the code is similar).

The script will traverse the original fisheye camera images in the corresponding folder, obtain the camera intrinsic parameters, and save them to output_mergepic_matrix\remapmatrix, such as calibration_data_front_1080p.yaml.

2.2. Obtain the perspective transformation matrix

Go to the \tools\get_matrix_map\get_Trans_matrix\ folder. Run the four 1000x1200_transfisheye_.py files.

The script will read the point-to-point mapping, obtain the perspective transformation mapping table, and save it to output_mergepic_matrix\remapmatrix, such as 1080p_transfor_front_1000x1200.yaml.

2.3. Establish the fusion weight table

Go to the \tools\get_mergepic\ folder. Run the four getmask_xx.py files, left-up corresponds to the top left fusion area.

The script will create a fusion weight image based on the set polygon fusion area, with the weight gradient changing according to pixel position, and save the weight images to \output_mergepic_matrix\mergepic, such as weights_left_down.png.

2.4. Modify the paths and macro commands in fisheye_simple.cpp to save the surround view images to the required folder. The default save location is \output_mergepic_matrix\AVM_PIC_LOG.

Replace the camera intrinsic matrix and perspective transformation matrix in the source code with the calculated matrix parameters.

For example, modify cv::Mat K_F, cv::Mat K2_F, cv::Mat D_F to the required camera intrinsic values.

Modify cv::Mat perspective_matrix_f, cv::Mat perspective_matrix_l, cv::Mat perspective_matrix_r, cv::Mat perspective_matrix_b to the required values. The code is in line 174.

In an actual project, replace the fusion images in AVM_code\source\merge with the images obtained in \output_mergepic_matrix\mergepic, such as weights_left_down.png.

2.5 Different projects have different resolutions, so pay attention to the pixel size modifications in the scripts for calculating the perspective transformation matrix and fusion images.

Modify the fisheye image, perspective transformation image, and surround view image sizes in fisheye_simple.cpp.

The code to be modified is as follows:

```
cpp复制代码::Point ori_point = {0, 0};
::Point point_left_up = {400, 400};
::Point point_right_up = {600, 400};
::Point point_left_down = {400, 800};
::Point point_right_down = {600, 800};
::Point AVM_point = {1000, 1200};

// Set the corresponding image sizes
int AVM_WIDTH = AVM_point.x;
int AVM_HIGHT = AVM_point.y;

// Original image size captured by the camera
int original_width = 1920;
int original_height = 1080;

// Undistorted image size (default four are the same size)
int undistorted_width = 1920;
int undistorted_height = 1080;
```

2.5 Recompile and run fisheye_simple.cpp.

#### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request


#### Gitee Feature

1.  You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
