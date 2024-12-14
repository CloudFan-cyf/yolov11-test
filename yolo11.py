import cv2
import time
import numpy as np
from ultralytics import YOLO

# 加载模型
model = YOLO('yolo11n-seg.pt')

# 输入视频路径
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频的宽、高和帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = "mangdao_1_processed_video.avi"

# 定义视频编码器和输出文件
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    success, frame = cap.read()

    if success:
        # 开始计时
        start = time.perf_counter()

        # 对帧进行推理
        results = model(frame)

        # 结束计时
        end = time.perf_counter()
        total_time = end - start
        fps_display = 1 / total_time

        # 获取第一个检测结果
        result = results[0]
        masks = result.masks  # 分割的 mask 信息
        if masks is not None:
            start_mask_time = time.perf_counter()
            # 将所有对象的掩码合并为单个掩码
            combined_mask = masks.data.sum(axis=0)  # 将多个掩码合并
            combined_mask = combined_mask.cpu().numpy()  # 转换为 NumPy 数组
            combined_mask = (combined_mask > 0).astype(np.uint8)  # 二值化

            # 确保掩码尺寸与帧尺寸一致
            combined_mask = cv2.resize(combined_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 应用掩码过滤未分割的区域
            filtered_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

            # 生成掩码的反掩码
            inverse_mask = cv2.bitwise_not(combined_mask * 255)  # 生成反掩码（范围为0-255）
            inverse_mask = (inverse_mask > 0).astype(np.uint8)  # 确保数据类型正确

            # 对反掩码区域进行边缘提取
            edges = cv2.Canny(frame, 100, 200)  # 使用Canny边缘检测
            edges = cv2.bitwise_and(edges, edges, mask=inverse_mask)  # 仅对反掩码区域提取边缘

            # 对边缘图像进行平滑处理
            blurred_edges = cv2.GaussianBlur(edges, (5, 5), 0)

            # 将平滑后的图像叠加到原图
            smooth_frame = frame.copy()
            smooth_frame[inverse_mask == 1] = cv2.cvtColor(blurred_edges, cv2.COLOR_GRAY2BGR)[inverse_mask == 1]

            # 合并过滤后的图像和去细节图像
            combined_result = cv2.addWeighted(filtered_frame, 0.7, smooth_frame, 0.3, 0)
            end_mask_time = time.perf_counter()
            total_mask_time = end_mask_time - start_mask_time+total_time
            fps_mask = 1 / total_mask_time



            # 显示合并后的图像
            cv2.putText(combined_result, f'FPS: {fps_mask:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # 保存处理后的帧到输出视频
            out.write(combined_result)
            cv2.imshow('Combined Result', combined_result)

        # 在帧上绘制检测结果
        annotated_frame = result.plot()
        cv2.putText(annotated_frame, f'FPS: {fps_display:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('YOLO', annotated_frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
