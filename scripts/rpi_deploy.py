import onnxruntime as ort
import numpy as np
import cv2
import time
from utils.post_process import PostProcessor, KalmanFilter
from configs.config import Config

class BEVDeployer:
    def __init__(self, model_path):
        # Initialize ONNX runtime session
        self.session = ort.InferenceSession(model_path)
        self.post_processor = PostProcessor(threshold=0.1)
        self.kf_x = KalmanFilter()
        self.kf_y = KalmanFilter()
        
    def preprocess(self, img_f, img_r):
        # Resize and normalize
        img_f = cv2.resize(img_f, (Config.img_w, Config.img_h))
        img_r = cv2.resize(img_r, (Config.img_w, Config.img_h))
        
        img_f = img_f.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_r = img_r.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Batch and Cam dimension
        img = np.stack([img_f, img_r], axis=0)[np.newaxis, ...] # (1, 2, 3, H, W)
        return img

    def run(self, img_f, img_r, calibs):
        # Preprocess
        img = self.preprocess(img_f, img_r)
        
        # Run inference
        inputs = {
            'img': img,
            'calibs': calibs.astype(np.float32)[np.newaxis, ...] # (1, 2, 3, 4)
        }
        
        start_time = time.time()
        outputs = self.session.run(None, inputs)
        inf_time = time.time() - start_time
        
        # Post-process
        pred_hm, pred_off = outputs
        # Convert back to torch for post-processor (or implement numpy version)
        import torch
        res = self.post_processor.decode_heatmap(torch.from_numpy(pred_hm), torch.from_numpy(pred_off))
        
        # Smooth with Kalman Filter
        smooth_dx = self.kf_x.update(res['delta_x'].item())
        smooth_dy = self.kf_y.update(res['delta_y'].item())
        
        return {
            'delta_x': smooth_dx,
            'delta_y': smooth_dy,
            'points': res['points_mm'],
            'inf_time': inf_time
        }

if __name__ == '__main__':
    # Usage example
    # deployer = BEVDeployer('bevformer.onnx')
    # img_f = cv2.imread('front.jpg')
    # img_r = cv2.imread('rear.jpg')
    # calibs = ...
    # result = deployer.run(img_f, img_r, calibs)
    # print(f"Offsets: ΔX={result['delta_x']:.2f}mm, ΔY={result['delta_y']:.2f}mm")
    print("Deployment script ready.")
