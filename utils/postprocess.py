import numpy as np
from filterpy.kalman import KalmanFilter as KF

def subpixel_peak(heatmap):
    h, w = heatmap.shape
    idx = np.argmax(heatmap)
    v0 = idx // w
    u0 = idx % w
    if u0 < 1 or u0 > w - 2 or v0 < 1 or v0 > h - 2:
        return float(u0), float(v0)
    f = heatmap
    gx = 0.5 * (f[v0, u0 + 1] - f[v0, u0 - 1])
    gy = 0.5 * (f[v0 + 1, u0] - f[v0 - 1, u0])
    dxx = f[v0, u0 + 1] - 2.0 * f[v0, u0] + f[v0, u0 - 1]
    dyy = f[v0 + 1, u0] - 2.0 * f[v0, u0] + f[v0 - 1, u0]
    dxy = 0.25 * (f[v0 + 1, u0 + 1] - f[v0 + 1, u0 - 1] - f[v0 - 1, u0 + 1] + f[v0 - 1, u0 - 1])
    H = np.array([[dxx, dxy], [dxy, dyy]], dtype=np.float32)
    g = np.array([gx, gy], dtype=np.float32)
    try:
        offset = -np.linalg.inv(H) @ g
    except np.linalg.LinAlgError:
        offset = np.zeros(2, dtype=np.float32)
    u = float(u0 + np.clip(offset[0], -1.0, 1.0))
    v = float(v0 + np.clip(offset[1], -1.0, 1.0))
    u = float(np.clip(u, 0, w - 1))
    v = float(np.clip(v, 0, h - 1))
    return u, v

def confidence_weighted_centroid(keypoints, confidences):
    kp = np.asarray(keypoints, dtype=np.float32)
    conf = np.asarray(confidences, dtype=np.float32)
    conf = np.clip(conf, 0.0, 1.0)
    s = np.sum(conf) + 1e-6
    cx = float(np.sum(kp[:, 0] * conf) / s)
    cy = float(np.sum(kp[:, 1] * conf) / s)
    return cx, cy

def bev_to_phys(u, v, x_min, y_min, res):
    x = float(u * res + x_min)
    y = float(v * res + y_min)
    return x, y

def compute_bias(container_centroid, spreader_centroid):
    dx = float(spreader_centroid[0] - container_centroid[0])
    dy = float(spreader_centroid[1] - container_centroid[1])
    return dx, dy

class KalmanFilter:
    def __init__(self, q_pos=1e-2, q_vel=1e-1, r_meas=1e-1):
        self.kf = KF(dim_x=4, dim_z=2)
        self.kf.x = np.zeros(4, dtype=np.float32)
        self.kf.F = np.eye(4, dtype=np.float32)
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]], dtype=np.float32)
        q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)
        self.kf.Q = q
        self.kf.R = np.diag([r_meas, r_meas]).astype(np.float32)
        self.kf.P = np.eye(4, dtype=np.float32)

    def predict(self, dt):
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], dtype=np.float32)
        self.kf.predict()
        return float(self.kf.x[0]), float(self.kf.x[1])

    def update(self, dx_raw, dy_raw):
        z = np.array([dx_raw, dy_raw], dtype=np.float32)
        self.kf.update(z)
        return float(self.kf.x[0]), float(self.kf.x[1])
