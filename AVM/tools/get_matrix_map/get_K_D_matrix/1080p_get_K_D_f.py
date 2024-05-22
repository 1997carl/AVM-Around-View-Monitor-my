#输入带有棋盘格的图像，脚本计算相机内参和畸变系数
import cv2
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
import yaml
CHECKERBOARD = (6,9)#修改行和列
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('D:/mypython/works/fisheye/step1_ub_fisheye/front/*.jpg')  #D:/mypython/works/fisheye/step1_ub_fisheye/front/front/ - frame at 0m41s.jpg
imgcnt=0
for fname in images:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    imgcnt+=1
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
        
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)

        # Display the image with corners
        output_dir = "D:/mypython/works/fisheye/step1_ub_fisheye/finding/"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join("D:/mypython/works/fisheye/step1_ub_fisheye/findcheekboard/", f"corners_{imgcnt}.jpg")
        cv2.imwrite(save_path, img)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)  # Display for 500 milliseconds
        
    else:
        print(fname, "findChessboardCorners failed")
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    

print("imgcnt:", imgcnt)
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.f(" + str(D.tolist()) + ")")
# 保存字典到YAML文件
calibration_data_f = {
    'K_f': K.tolist(),
    'D_f': D.tolist(),
    "size":str(_img_shape[::-1])
}
output_file = 'D:\\mypython\\works\\fisheye\\output\\calibration_data_front_1080p.yaml' 
with open(output_file, 'w') as yaml_file:
    yaml.dump(calibration_data_f, yaml_file)

print(f"Calibration data saved to {output_file}")


panorama = cv2.imread('D:/mypython/works/fisheye/input/original_img/distorted_f.jpg') #D:/mypython/works/fisheye/measure_img/original_img/distorted_front.jpg

############################################### 改成所需要的定义鱼眼矫正的参数########################################################################## 修改参数
# K = np.array([[288.3808623369766, 0.0, 315.213405939198], [0.0, 288.0133993027946, 235.94171571751434], [0.0, 0.0, 1.0]])  # 鱼眼内参矩阵
# K2 = np.array([[288.3808623369766/3, 0.0,300], [0.0, 288.0133993027946/3, 235.94171571751434], [0.0, 0.0, 1.0]])  # 相机内参矩阵
# D = np.array([-0.1984224368952329, 0.009214061353032866, 0.010967338593001865, -0.0025246210528729876])  # 畸变系数

# K_b = np.array([[285.07509622671915, 0.0, 330.8860900425384], [0.0, 285.72178364115206, 259.83738661485916], [0.0, 0.0, 1.0]])  # 鱼眼内参矩阵- -
# K2_b = np.array([[285.07509622671915/3, 0.0, 330.8860900425384], [0.0, 285.72178364115206/3, 259.83738661485916], [0.0, 0.0, 1.0]])  # 相机内参矩阵
# D_b = np.array([-0.15819418803724894, -0.0485156240327049, 0.06027793980864648, -0.01927339703359863])  # 畸变系数
#######右侧相机去畸变内参##########

K_b = K.copy()
K2_b = K.copy()
K2_b[:1,:1]=K[:1,:1]/3
K2_b[1:2,1:2]=K[1:2,1:2]/3
D_b = D.copy()
tx = 0# x 方向的平移像素数
translation_matrix = np.array([[1, 0, tx], [0, 1, 0], [0, 0, 1]],dtype=np.float32)

# 进行鱼眼矫正
width, height = panorama.shape[:2]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_b, D_b, translation_matrix, K2_b, (int(height),int(width)), cv2.CV_16SC2)#修改为对应的参数

undistorted_panorama = cv2.remap(panorama, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#cv2.imshow('Image', undistorted_panorama)
# 定义裁剪区域

# x1, y1 = 0,0 # 左上角坐标
# x2, y2 = 700,1080   # 右下角坐标

# 进行裁剪
planar_image = undistorted_panorama
cv2.imshow('undistorted', planar_image)
print("按下任意键保存图片")
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('D:/mypython/works/fisheye/step_2undistored/front/output/undistorted_front.jpg', planar_image)#修改保存路径D:\mypython\works\fisheye\measure_img\original_img\distorted_front.jpg
#前视
# DIM=(640, 480)
# K=np.array([[284.99502635362364, 0.0, 300.27828654007135], [0.0, 284.5186362068991, 238.4126262617839], [0.0, 0.0, 1.0]])
# D=np.array([[-0.1754512266896831], [-0.00046628297758875045], [0.006797721272022029], [0.0005011272368661744]])

#左视
