import cv2
import numpy as np
import yaml
# Global variables
###################################### Load an image 载入图像
print("get transimg")
image = cv2.imread(
    "D:\\mypython\\works\\fisheye\\step1_ub_fisheye\\output\\undistorted_back.jpg"#D:\mypython\works\fisheye\step_2undistored\front\undistored_front.jpg
)  
if image is None:#确认图像是否载入成功
    print("Error: Unable to load the image.")
    exit()
cv2.imshow("Original Image", image)
print(" 显示原始图片")

size = (1000, 400)  # 变换后前视图像大小
# 定义前侧棋盘格位置
rectangle1_x = 260
rectangle1_y = 180
width1=120
height1=120

width2=120
height2=120
rectangle2_x = 620
rectangle2_y = 180

# Wait for the user to click four points
src_points = np.array(
    [
        (777,454),
        (868, 458),
        (724,521),
        (837,527),
        
        (1047,468),
        (1133,472),
        (1070,540),
        (1181,546),
    ],
    dtype=np.float32,
) #原始图片的像素位置

dst_points = np.array(
    [
        (0 + rectangle1_x, 0 + rectangle1_y),
        (width1 + rectangle1_x, 0 + rectangle1_y),
        (0 + rectangle1_x, height1 + rectangle1_y),
        (width1 + rectangle1_x, height1 + rectangle1_y),
        
        (0 + rectangle2_x, 0 + rectangle2_y),
        (width2 + rectangle2_x, 0 + rectangle2_y),
        (0 + rectangle2_x, height2 + rectangle2_y),
        (width2 + rectangle2_x, height2 + rectangle2_y),
    ],
    dtype=np.float32,
)  #变换后的像素位置  （156，190）  （550，156）

# 左侧棋盘格位置
# translation_x = 338
# translation_y = 46
# dst_points = np.array([(0 + translation_x, 0 + translation_y),
#                        (296+ translation_x, 0 + translation_y),
#                        (296+ translation_x, 124+ translation_y),     （156，190）  （550，156）
#                        (0 + translation_x, 124+ translation_y)], dtype=np.float32)

# #后侧棋盘格位置
# translation_x = 238
# translation_y = 40
# dst_points = np.array([(0 + translation_x, 0 + translation_y),
#                        (112+ translation_x, 0 + translation_y),
#                        (112+ translation_x, 80+ translation_y),
#                        (0 + translation_x, 80+ translation_y)], dtype=np.float32)

# transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
transform_matrix, mask = cv2.findHomography(
    src_points,
    dst_points,
    method=None,
    ransacReprojThreshold=None,
    mask=None,
    maxIters=None,
    confidence=None,
)

cv2.imwrite("D:/mypython/works/fisheye/step3_get_trans/output/before_transformed_image_b.jpg", image)
transformed_image = cv2.warpPerspective(image, transform_matrix, size)
cv2.imwrite("D:/mypython/works/fisheye/step3_get_trans/output/transformed_image_b.jpg", transformed_image)

# 显示图像
if transformed_image is not None:
    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("")

############ 打印透视变换矩阵，得到透视变换矩阵。
print("后视透视变换矩阵:")
print(transform_matrix)
# 保存字典到YAML文件
calibration_data_r = {
    'transform_matrix_left': transform_matrix.tolist(),
}

output_file = 'D:\\mypython\\works\\fisheye\\output\\1080p_transfor_back_1000x1200.yaml'
with open(output_file, 'w') as yaml_file:
    yaml.dump(calibration_data_r, yaml_file, default_flow_style=True)
    # 将每行前面的 '-' 符号去除
#break

cv2.destroyAllWindows()