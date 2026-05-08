import numpy as np
import yaml

class Calibration:
    def __init__(self, calib_file):
        with open(calib_file, 'r') as f:
            data = yaml.safe_load(f)
        
        self.front = self._parse_cam(data['cameras']['front_camera'])
        self.rear = self._parse_cam(data['cameras']['rear_camera'])

    def _parse_cam(self, cam_data):
        K = np.array(cam_data['K']).reshape(3, 3)
        D = np.array(cam_data['D'])
        R = np.array(cam_data['extrinsic']['R']).reshape(3, 3)
        T = np.array(cam_data['extrinsic']['T']).reshape(3, 1)
        
        # P = K * [R | T]
        # Extrinsic matrix [R | T]
        ext = np.hstack([R, T])
        P = K @ ext
        
        return {
            'K': K,
            'D': D,
            'R': R,
            'T': T,
            'ext': ext,
            'P': P
        }

    def project_3d_to_bev(self, points_3d, x_range, y_range, resolution):
        """
        Convert 3D points (X, Y, Z) to BEV grid coordinates (u, v)
        Assume BEV is XY plane.
        """
        u = (points_3d[:, 0] - x_range[0]) / resolution
        v = (points_3d[:, 1] - y_range[0]) / resolution
        return np.stack([u, v], axis=1)

    def back_project_2d_to_3d(self, kp_2d, cam_type='front', z_world=0):
        """
        Back-project 2D image points to 3D world points assuming they lie on a plane Z = z_world.
        kp_2d: (N, 2)
        cam_type: 'front' or 'rear'
        """
        cam = self.front if cam_type == 'front' else self.rear
        K_inv = np.linalg.inv(cam['K'])
        R = cam['R']
        T = cam['T']
        
        # Camera space ray: r_c = K^-1 * [u, v, 1]^T
        # World space point: P_w = R^-1 * (s * r_c - T)
        # We need to find 's' such that Z_w = z_world
        
        R_inv = R.T
        T_inv = -R.T @ T
        
        # P_w = s * (R_inv @ r_c) + T_inv
        # Z_w = s * (R_inv @ r_c)[2] + T_inv[2] = z_world
        # s = (z_world - T_inv[2]) / (R_inv @ r_c)[2]
        
        kp_2d_hom = np.hstack([kp_2d, np.ones((len(kp_2d), 1))])
        rays_c = (K_inv @ kp_2d_hom.T).T
        
        points_3d = []
        for r_c in rays_c:
            r_w = R_inv @ r_c
            s = (z_world - T_inv[2]) / (r_w[2] + 1e-6)
            p_w = s * r_w + T_inv.squeeze()
            points_3d.append(p_w)
            
        return np.array(points_3d)
