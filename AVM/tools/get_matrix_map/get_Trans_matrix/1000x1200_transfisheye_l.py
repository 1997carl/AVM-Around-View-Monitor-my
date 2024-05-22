import cv2
import numpy as np
import yaml
import xml.dom.minidom as minidom
# Global variables
###################################### Load an image 载入图像
import xml.etree.ElementTree as ET
# 解析XML文件
tree = ET.parse("D:\\myc++\\c++_works\\get_cam_paramater\\result_text.xml")
root = tree.getroot()

# 存储结果
result = []

# 遍历所有world_point子节点
for point in root.findall("./imgpoint/_"):
    # 获取x和y坐标
    x = float(point.find("x").text)
    y = float(point.find("y").text)
    
    
    
    # 将坐标添加到结果中
    result.append((x, y))

# 输出结果
print("world_point坐标数据：")
i=0
for point in result:
    i=i+1
    print("point",i,":(", point[0], point[1],")")
input("Press Enter to continue...")
cv2.waitKey(10)

src_points=np.array(result)
print(src_points)



print("get transimg")
image = cv2.imread(
    "D:\\mypython\\works\\fisheye\\step1_ub_fisheye\\output\\undistored_left.jpg"#D:\mypython\works\fisheye\step_2undistored\front\undistored_front.jpg
)  # D:\mypython\fisheye\measure_img\undistoreted\result_image.jpg
if image is None:#确认图像是否载入成功
    print("Error: Unable to load the image.")
    exit()
cv2.imshow("Original Image", image)
print(" 显示原始图片")

size = (1200, 400)  # 变换后前视图像大小
# 定义前侧棋盘格位置
rectangle1_x = 180
rectangle1_y = 260
width1=120
height1=120

width2=120
height2=120
rectangle2_x = 900
rectangle2_y = 260


# src_points = np.array(
#     [
#         ( 662.2052001953125, 540.5298461914062),
#         (1215.3515625, 518.4324340820312 ),
#         (557.5762939453125, 634.1400146484375),
#         (1279.95361328125, 582.0697631835938  ),
        
#     ],
#     dtype=np.float32,
# ) #原始图片的像素位置


boardHeight = 4
boardWidth = 20
squareSize = 40

dst_points = []
for j in range(boardHeight):
    for k in range(boardWidth):
        dst_points.append((220.0 + k * squareSize, 220.0 + j * squareSize))

for i, corner in enumerate(dst_points):
    print(f"dst_points {i+1}: ({corner[0]}, {corner[1]})")
    
dst_points = np.array(dst_points)
# dst_points = np.array(
#     [
#         (0 + rectangle1_x, 0 + rectangle1_y),
#         (width1 + rectangle1_x, 0 + rectangle1_y),
#         (0 + rectangle1_x, height1 + rectangle1_y),
#         (width1 + rectangle1_x, height1 + rectangle1_y),
        
#         (0 + rectangle2_x, 0 + rectangle2_y),
#         (width2 + rectangle2_x, 0 + rectangle2_y),
#         (0 + rectangle2_x, height2 + rectangle2_y),
#         (width2 + rectangle2_x, height2 + rectangle2_y),
#     ],
#    dtype=np.float32,
#)  #变换后的像素位置  （156，190）  （550，156）


transform_matrix, mask = cv2.findHomography(
    src_points,
    dst_points,
    method=0  ,
    ransacReprojThreshold=10,
    mask=None,
    maxIters=200,
    confidence=0.2,
)
print(transform_matrix)

cv2.imwrite("D:/mypython/works/fisheye/step3_get_trans/output/before_transformed_image_l.jpg", image)
transformed_image = cv2.warpPerspective(image, transform_matrix, size)
cv2.imshow("Transformed Image3", transformed_image)
cv2.imwrite("D:/mypython/works/fisheye/step3_get_trans/output/transformed_image_l.jpg", transformed_image)

# 显示图像
if transformed_image is not None:
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("")

############ 打印透视变换矩阵，得到透视变换矩阵。
print("前视透视变换矩阵:")
print(transform_matrix)
# 保存字典到YAML文件
calibration_data_r = {
    'transform_matrix_left': transform_matrix.tolist(),
}

output_file = 'D:\\mypython\\works\\fisheye\\output\\1080p_transfor_left_1000x1200.yaml'
with open(output_file, 'w') as yaml_file:
    yaml.dump(calibration_data_r, yaml_file, default_flow_style=True)
    # 将每行前面的 '-' 符号去除
#break

















cv2.destroyAllWindows()


#  显示原始图片
# Clicked at: (128, 265)
# Clicked at: (189, 265)
# Clicked at: (46, 298)
# Clicked at: (135, 299)
# Clicked at: (358, 273)
# Clicked at: (385, 272)
# Clicked at: (378, 300)
# Clicked at: (417, 301)
# 5选点完成
# 前视透视变换矩阵:
# [[-5.94242238e-01 -2.09997560e+00  5.88672109e+02]
#  [ 2.09156953e-02 -1.55912599e+00  3.69004197e+02]
#  [ 9.29970428e-05 -5.26084911e-03  1.00000000e+00]]

# [[-5.94242238e-01, -2.09997560e+00 , 5.88672109e+02,2.09156953e-02 ,-1.55912599e+00 , 3.69004197e+02,9.29970428e-05 ,-5.26084911e-03 , 1.00000000e+00]]
