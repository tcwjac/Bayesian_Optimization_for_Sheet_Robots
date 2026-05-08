from skimage.morphology import skeletonize
from skimage import img_as_bool
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import gaussian_filter1d

numoflayer = '3layer'
ratio = '_1to1'
kuadu = '_3mm'
frequency = '_1Hz'
cichangqiangdu = '_30mT'

y_min = 200
y_max = 660
x_min = 157
x_max = 664

y_length = y_max-y_min
x_length = x_max-x_min

binary_threshold = 47


'调整尺寸和二值化阈值并提取中心线，且找到原点与放缩比例'
filename = r"C:\Users\xchenim\Documents\GitHub\Bayesian_Optimization_for_Sheet_Robots\19_7_Excel\test10\Ecoflex_3_0_1.bmp"
# filename = "D:\\biaozhengshiyan2024-04-25\\2zhouqi_3mmx12mm_2Hz_max_final_2024-05-04_C001H001S0001\\biaochi.png"
ori = cv2.imread(filename)
ori_img = cv2.imread(filename)[y_min:y_max, x_min:x_max]
img = cv2.cvtColor(ori_img,cv2.COLOR_BGR2GRAY) 

'二值化图像处理'
_, img = cv2.threshold(img,binary_threshold, 255, cv2.THRESH_BINARY_INV)

'膨胀腐蚀'
# kernel = np.ones(shape=(5, 5))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
img = cv2.dilate(img,kernel)
img = cv2.dilate(img,kernel)
img = cv2.dilate(img,kernel)
img = cv2.dilate(img,kernel)
img = cv2.dilate(img,kernel)
img = cv2.dilate(img,kernel)
img = cv2.dilate(img,kernel)
img = cv2.dilate(img,kernel)

# img = cv2.erode(img,kernel)        #腐蚀图像
# img = cv2.erode(img,kernel)  
# img = cv2.erode(img,kernel)  
# img = cv2.erode(img,kernel)  

# y_length = len(img)
# x_length = len(img[0])
# print(x_length)

# img = cv2.dilate(img,kernel)



contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
blank_image = np.zeros_like(img)
cv2.drawContours(blank_image, [largest_contour], -1, 255, thickness=cv2.FILLED)
bool_image = img_as_bool(blank_image)
skeletonized_image = skeletonize(bool_image)


'可视化'
overlay_image = np.stack([255-np.zeros_like(img)] * 3, axis=-1) 
# overlay_image = np.zeros_like(np_image)
overlay_image[np.where(skeletonized_image)] = [0, 0, 0]

# print(skeletonized_image)
X = []
Y = []
for i in range(len(overlay_image)):
    for j in range(len(overlay_image[0])):
        if overlay_image[i][j][0] == 0 and overlay_image[i][j][1] == 0 and overlay_image[i][j][2] == 0:
            Y.append(i+y_min)
            X.append(j+x_min)
            # print(str(j)+','+str(i))
            # print()

        
for i in range(len(X)):
    print(str(X[i])+'。'+str(Y[i]))
# plt.scatter(X,Y)
# print(overlay_image[57][416])
# print(X)
# print(overlay_image[56][])
# print(len(overlay_image))
# cv2.imshow('result',contours)
# cv2.imshow('result',img)
# # cv2.imshow('result',overlay_image)
# cv2.waitKey()
# cv2.destroyAllWindows()

# contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 创建边界图像
# boundary_mask = np.zeros_like(img)
# cv2.drawContours(boundary_mask, contours, -1, 255, 1)

# # 提取下边界：在每列上找边界点中y值最大的点
# h, w = img.shape
# # lower_boundary_y = np.full(w, -1, dtype=int)
# X = []
# Y = []
# for x in range(w):
#     ys = np.where(boundary_mask[:, x] == 255)[0]
#     if len(ys) > 0:
#         X.append(x)
#         Y.append(ys.max()-5)
#         # lower_boundary_y[x] = ys.max()
# # print(lower_boundary_y)


for i in range(len(X)):
    y = Y[i]
    if y != -1:
        ori[Y[i], X[i]] = [0, 0, 255]  # 红色表示下边界

# 显示结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# plt.title("Lower Boundary Overlay")
plt.imshow(ori[..., ::-1])
plt.axis('on')

# plt.gca().invert_yaxis()
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.show()

# plt.plot(X,Y, color='red')
# # plt.gca().invert_yaxis()
# # plt.tight_layout()
# plt.show()