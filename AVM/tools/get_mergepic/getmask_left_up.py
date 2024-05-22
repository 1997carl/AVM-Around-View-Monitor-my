import os
import numpy as np
import cv2
from PIL import Image
# 读取两张图片，建立重叠区域大的权值表
def get_outmost_polygon_boundary(img):#传入图片 得到边界
    """
    Given a mask image with the mask describes the overlapping region of
    two images, get the outmost contour of this region.
    """
    mask = get_mask(img)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
    cnts, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # get the contour with largest aera
    C = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]
    # polygon approximation
    polygon = cv2.approxPolyDP(C, 0.009 * cv2.arcLength(C, True), True)

    return polygon

def get_mask(img):
    """
    Convert an image to a mask array.
    """
    #smoothed_img = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转为单通道灰度图片
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)#二值化，将灰度图变成黑白图
    return mask

def get_overlap_region_mask(imA, imB):#输入相同大小的图片计算重叠的区域，输出重叠部分的掩码
    """
    Given two images of the save size, get their overlapping region and
    convert this region to a mask array.
    """
    overlap = cv2.bitwise_and(imA, imB)#位与操作，得到两个图片的位与操作
    mask = get_mask(overlap)
    mask = cv2.dilate(mask,np.ones((2, 2),np.uint8),iterations=2)#在这个上下文中，膨胀操作通常用于扩展掩码中的白色区域，以便更好地捕捉和处理重叠的区域。
    return mask


def clip(img,x, y, w, h ):
    img_clip=img[y:y+h, x:x+w]
    return img_clip 
    
def main():
    #STEP1 裁剪待融合区域的两幅图片
    #读取原始图片
    front = cv2.imread('D:/mypython/works/get_mask_and _weights/source/400x400/left_up_f.jpg')
    left=cv2.imread('D:/mypython/works/get_mask_and _weights/source/400x400/left_up_l.jpg')
    #获取图片尺寸信息
    front_height, front_width, front_channels = front.shape
    left_height, left_width, left_channels = left.shape
    #截取图片，获得重叠区域的图片
    front_overlap_left = clip(front,0,0,left_width,front_height)  
    left_overlap_front = clip(left,0,0,left_width,front_height)
    ###########直接传入重叠区域的图片##########
    front_overlap_left = cv2.imread('D:/mypython/works/get_mask_and _weights/source/400x400/left_up_f.jpg')
    left_overlap_front=cv2.imread('D:/mypython/works/get_mask_and _weights/source/400x400/left_up_l.jpg')
    #step2 生成边界
    #2.1图片中重叠的掩码区域
    left_up_overlap=get_overlap_region_mask(front_overlap_left,left_overlap_front)#重叠区域掩码 用于后续布尔运算得到边界区域
    left_up_overlapInv = cv2.bitwise_not(left_up_overlap)#非重叠部分的掩码
    front_overlap_left_diff = cv2.bitwise_and(front_overlap_left, front_overlap_left, mask=left_up_overlapInv)#图像a非重叠的部分
    left_overlap_front_diff = cv2.bitwise_and(left_overlap_front, left_overlap_front, mask=left_up_overlapInv)#图像b非重叠的部分
    
    # cv2.imshow('front_without_ovelap', front_overlap_left_diff)#即用于得到边界区域的图片
    # cv2.imshow('left_without_ovelap', left_overlap_front_diff)
    # cv2.imshow('left_up_overlap', left_up_overlap)
    #2.2初始化权重图片的像素值，全部置为1
    shape = front_overlap_left.shape
    G = np.ones(shape, dtype=np.float32)
    #2.3得到图像a区域的掩码，初始的掩码区域
    polyA = get_outmost_polygon_boundary(front_overlap_left_diff)#前视图非重叠的边界
    polyB = get_outmost_polygon_boundary(left_overlap_front_diff)#左视图非重叠的边界
    
    #####################################绘制边界的代码############################
    #创建空白图像
    width = 500
    height = 500
    channels = 3
    img = np.zeros((height, width, channels), dtype=np.uint8)

    # 定义多边形的顶点列表
    polygon = polyB
    # 在图像上绘制多边形
    color = (0, 0, 255)  # BGR格式颜色值
    thickness = 2
    isClosed = True
    cv2.polylines(img, [polygon], isClosed, color, thickness)

    # 显示绘制后的图像
    # cv2.imshow("image with polygon", img)
    cv2.waitKey(0)
    for vertex in polyB:
        x, y = vertex[0]
        print(f"Vertex: ({x}, {y})")
#########################################结束绘制边界######################

    #####################################绘制边界的代码############################
    #创建空白图像
    width = 500
    height = 500
    channels = 3
    img = np.zeros((height, width, channels), dtype=np.uint8)

    # 定义多边形的顶点列表
    polygon = polyA
    # 在图像上绘制多边形
    color = (0, 0, 255)  # BGR格式颜色值
    thickness = 2
    isClosed = True
    cv2.polylines(img, [polygon], isClosed, color, thickness)

    # 显示绘制后的图像
    # cv2.imshow("image with polygonA", img)
    cv2.waitKey(0)
    for vertex in polyA:
        x, y = vertex[0]
        print(f"Vertex: ({x}, {y})")
#########################################结束绘制边界######################

    cv2.waitKey(0)
    #3生成权重图片
    #3.1开始初始化权重图片，生成需要遍历的坐标
    height, width, channels = front_overlap_left.shape#需要修改大小
    mask = np.ones((height, width), dtype=int)
    indices = np.where(mask)#循环的坐标点
    dist_threshold=5#一个边界距离的阈值
    unzipped_indices = list(zip(*indices))
    num_tuples = len(unzipped_indices)
    print("indices=",num_tuples)
    num_for=0
    #3.2遍历像素点位置，建立权值图
    
    
    
    
    
    
    
    
    
    
    polyB=np.array([[0, 300], [0, 400], [400, 400]], dtype=np.int32)
    polyA=np.array([[300, 0], [400, 0], [400, 400]], dtype=np.int32)
    polygon_points = []
    for y in range(height):
        for x in range(width):
            if cv2.pointPolygonTest(polyB, (x, y), False) < 0 and cv2.pointPolygonTest(polyB, (x, y), False) < 0:
                polygon_points.append((y, x))

# # 遍历多边形内部的坐标
# for y, x in polygon_points:
#     mask[y, x] = 1
#     cv2.fillPoly(G, [polyA], 1)
#     cv2.fillPoly(G, [polyB], 0)
    mask = np.ones((height, width), dtype=np.uint8)

    # 定义两个多边形区域的坐标（这里以矩形为例）

    # 使用 fillPoly 将两个多边形区域填充为1
    cv2.fillPoly(mask, [polyB], 0)

    # 将图像的每个通道与 mask 相乘
    G = G * mask[:, :, np.newaxis]
    

    
    # for y, x in zip(*indices):  #将系列的纵坐标点，解包成为x和y的元组
    for y, x in polygon_points:
        num_for=num_for+1
        #convert this x,y int an INT tuple
        xy_tuple = tuple([int(x), int(y)])
        distToB = cv2.pointPolygonTest(polyB, xy_tuple, True)
        distToA = cv2.pointPolygonTest(polyA, xy_tuple, True)
        print("distToBnormal",distToB)
        # if  (distToA>=0):
        #     G[y, x]=1
        if (distToB < dist_threshold):    #>0在边界内部，=0在边界上，<0 在边界外部
            distToB *= distToB
            distToA *= distToA 
            G[y, x] = distToB / (distToA + distToB)         #G 权重值，left_up_overlap ，共同重叠部分
            #print("渐变区域")
        # else:#不在多边形区域内时，赋值为1
        #     G[y, x]=0
        #     #print("值为0")

        Image.fromarray((G * 255).astype(np.uint8)).save("D:/mypython/works/get_mask_and _weights/output/weights_left_up.png")#保存权重矩阵

        #Image.fromarray(left_up_overlap.astype(np.uint8)).save("D:/mypython/works/get_mask_and _weights/output/masks_change.png")#掩码矩阵
        print("loadng{}%,y:{},x:{},dis{},G[y, x]:{}".format(num_for/num_tuples*100,y,x,distToB,G[y, x]))
main()