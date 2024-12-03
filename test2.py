# test.ipynb

# 导入所需的库
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 辅助函数：绘制特征点
def draw_keypoints(image, keypoints, color=(0, 255, 0)):
    keypoints_img = np.copy(image)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])  # 使用 int() 将坐标转换为整数
        cv2.circle(keypoints_img, (x, y), 7, color, -1)
    return keypoints_img

# 辅助函数：去除黑边
def remove_black_border(warped_image):
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    cropped_image = warped_image[y:y+h, x:x+w]
    return cropped_image

# 图像预处理、特征点提取、匹配和拼接
def stitch_images(img1_path, img2_path):
    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测特征点和计算描述符
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # 使用FLANN匹配器进行特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 存储好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 至少需要4个匹配点来计算单应性矩阵
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # 计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 使用透视变换将图像进行拼接
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        # 计算拼接后图像的尺寸
        new_width = width1 + width2
        new_height = max(height1, height2)

        warped_image = cv2.warpPerspective(img1, H, (new_width, new_height))

        # 计算 img2 在 warped_image 中的放置位置
        # 确保 img2 在 warped_image 的安全区域内
        warped_image[0:height2, 0:width2] = img2

        # 去除黑边
        stitched_image = remove_black_border(warped_image)

        # 计算处理时间并输出帧率
        start_time = time.time()
        end_time = time.time()
        frame_rate = 1 / (end_time - start_time)
        print(f"Processing time: {end_time - start_time:.4f} seconds, Frame Rate: {frame_rate:.2f} FPS")

        # 绘制特征点并展示结果
        keypoints_img1 = draw_keypoints(img1, keypoints1)
        keypoints_img2 = draw_keypoints(img2, keypoints2)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Image 1 Keypoints')
        plt.imshow(cv2.cvtColor(keypoints_img1, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Image 2 Keypoints')
        plt.imshow(cv2.cvtColor(keypoints_img2, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Stitched Image')
        plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.show()

        return stitched_image
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 4))
        return None

# 调用函数进行图像拼接
stitch_images('/home/shijian/桌面/视觉测试/3.jpg', '/home/shijian/桌面/视觉测试/4.jpg')
