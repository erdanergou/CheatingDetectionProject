import time

import cv2
import torch

from estimator.pose_estimator import AlphaPoseEstimator
from estimator.yolo_detector import YoloV5Detector
from utils.vis import draw_keypoints136

yolov5_weight = 'weights/yolov5s.torchscript.pt'
alphapose_weight = 'weights/halpe136_mobile.torchscript.pth'
box_color = (0, 255, 0)

torch._C._jit_set_profiling_mode(False)
torch.jit.optimized_execution(False)

if __name__ == '__main__':
    detector = YoloV5Detector(weights=yolov5_weight, device='cuda')
    pose = AlphaPoseEstimator(weights=alphapose_weight, device='cuda')

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
    i = 0
    while ret:
        last_time = time.time()
        pred = detector.detect(frame)
        preds_kps, preds_scores = pose.estimate(frame, pred)
        if preds_kps.shape[0] > 0:
            draw_keypoints136(frame, preds_kps, preds_scores)
        current_time = time.time()
        fps = 1 / (current_time - last_time)

        cv2.putText(frame, "FPS: %.2f" % fps, (0, 52), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow("yolov5", frame)
        # 下一帧
        ret, frame = cap.read()
        if cv2.waitKey(30) and 0xFF == 'q':
            break
    cap.release()
    cv2.destroyAllWindows()
