import os
import json
import shutil
import yaml
import numpy as np
import cv2
import re

# ==========================================
# 参数配置区
# ==========================================
CONFIG = {
    # 路径配置
    "raw_front_dir": "data/raw/images/front",
    "raw_rear_dir": "data/raw/images/rear",
    "pair_mapping_file": "data/pair_mapping.json",
    "anno_file": "data/annotations/_annotations.coco.json",
    "calib_file": "data/raw/camera_calib.yaml.txt",
    "output_dir": "data/processed",
    
    # BEV 参数
    "bev_x_range": [-750, 750],  # mm
    "bev_y_range": [0, 600],     # mm
    "bev_resolution": 2.5,       # mm/pixel
    
    # 热图参数
    "sigma": 2.0,
    "num_keypoints": 8,
    # 最大允许前后相机时间戳差，超过则丢弃该配对
    "max_pair_gap": 1600000
}

TIMESTAMP_PATTERN = re.compile(r'(\d+)')

def extract_timestamp(filename):
    m = TIMESTAMP_PATTERN.findall(filename)
    if not m:
        return None
    # use the last numeric group as timestamp
    return int(m[-1])

def pair_by_nearest_timestamp(front_files, rear_files, max_gap=None):
    """
    Pair each front image with the nearest rear timestamp (one-to-one).
    """
    front_meta = []
    rear_meta = []
    for f in front_files:
        ts = extract_timestamp(f)
        if ts is not None:
            front_meta.append((f, ts))
    for f in rear_files:
        ts = extract_timestamp(f)
        if ts is not None:
            rear_meta.append((f, ts))

    front_meta.sort(key=lambda x: x[1])
    rear_meta.sort(key=lambda x: x[1])

    used_rear = set()
    pairs = []
    for ff, fts in front_meta:
        best_idx = -1
        best_gap = None
        for i, (_, rts) in enumerate(rear_meta):
            if i in used_rear:
                continue
            gap = abs(fts - rts)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_idx = i
        if best_idx >= 0:
            if max_gap is not None and best_gap is not None and best_gap > max_gap:
                continue
            used_rear.add(best_idx)
            pairs.append((ff, rear_meta[best_idx][0], best_gap))
    return pairs

def load_explicit_pairs(mapping_file, front_files, rear_files):
    if not os.path.exists(mapping_file):
        return None
    with open(mapping_file, "r") as f:
        data = json.load(f)
    pairs = data.get("pairs", [])
    if not isinstance(pairs, list):
        return None
    front_set = set(front_files)
    rear_set = set(rear_files)
    valid = []
    for x in pairs:
        if not isinstance(x, dict):
            continue
        ff = x.get("front")
        rr = x.get("rear")
        if ff in front_set and rr in rear_set:
            valid.append((ff, rr, x.get("gap", None)))
    return valid

def save_mapping_template(mapping_file, pairs):
    os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
    payload = {
        "description": "Front-to-rear explicit pairing. Edit rear for each front if needed.",
        "pairs": [{"front": f, "rear": r, "gap": int(g) if g is not None else None} for f, r, g in pairs]
    }
    with open(mapping_file, "w") as f:
        json.dump(payload, f, indent=2)

def load_calib(yaml_path):
    """读取相机标定文件"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    def parse_cam(cam_data):
        return {
            'K': np.array(cam_data['K']).reshape(3, 3),
            'R': np.array(cam_data['extrinsic']['R']).reshape(3, 3),
            'T': np.array(cam_data['extrinsic']['T']).reshape(3, 1)
        }
        
    return {
        'front': parse_cam(data['cameras']['front_camera']),
        'rear': parse_cam(data['cameras']['rear_camera'])
    }

def back_project_to_ground(u, v, K, R, T, Z=0):
    """
    将像素坐标 (u, v) 反投影到物理世界坐标系中的 Z=0 平面上。
    公式:
    s * [u, v, 1]^T = K * (R * [X, Y, Z]^T + T)
    => s * K^-1 * [u, v, 1]^T = R * [X, Y, Z]^T + T
    => R^-1 * (s * K^-1 * [u, v, 1]^T - T) = [X, Y, Z]^T
    """
    K_inv = np.linalg.inv(K)
    R_inv = np.linalg.inv(R)
    
    # 射线在相机坐标系下的方向
    ray_c = K_inv @ np.array([[u], [v], [1.0]])
    
    # R^-1 * ray_c
    dir_w = R_inv @ ray_c
    # R^-1 * T
    origin_w = R_inv @ T
    
    # [X, Y, Z]^T = s * dir_w - origin_w
    # 令 Z = 0: s * dir_w[2] - origin_w[2] = 0 => s = origin_w[2] / dir_w[2]
    # 注意: T是世界到相机的平移，因此原点是 -R^-1 * T
    
    s = origin_w[2, 0] / dir_w[2, 0]
    point_w = s * dir_w - origin_w
    
    return point_w[0, 0], point_w[1, 0]

def physical_to_bev(x, y, x_range, y_range, resolution):
    """将物理坐标 (X, Y) 转换为 BEV 像素坐标 (u, v)"""
    u = (x - x_range[0]) / resolution
    v = (y - y_range[0]) / resolution
    return u, v

def generate_gaussian_heatmap(size, points, sigma):
    """
    生成多通道高斯热图。
    size: (H, W)
    points: list of (u, v) 浮点坐标
    sigma: 高斯核标准差
    """
    H, W = size
    num_points = len(points)
    heatmap = np.zeros((num_points, H, W), dtype=np.float32)
    
    for i, (u, v) in enumerate(points):
        # 边界检查 (允许稍微越界以画出部分高斯核)
        if u < -3*sigma or u >= W + 3*sigma or v < -3*sigma or v >= H + 3*sigma:
            continue
            
        # 计算边界
        ul = int(np.floor(u - 3 * sigma))
        br = int(np.ceil(u + 3 * sigma))
        ut = int(np.floor(v - 3 * sigma))
        bb = int(np.ceil(v + 3 * sigma))
        
        # 限制在图像范围内
        ul = max(0, ul)
        br = min(W - 1, br)
        ut = max(0, ut)
        bb = min(H - 1, bb)
        
        # 生成网格
        x = np.arange(ul, br + 1, 1, dtype=np.float32)
        y = np.arange(ut, bb + 1, 1, dtype=np.float32)
        y, x = np.meshgrid(y, x, indexing='ij')
        
        # 计算高斯分布
        g = np.exp(-((x - u)**2 + (y - v)**2) / (2 * sigma**2))
        
        # 将高斯分布叠加到热图上 (取最大值)
        heatmap[i, ut:bb+1, ul:br+1] = np.maximum(heatmap[i, ut:bb+1, ul:br+1], g)
        
    return heatmap

def process_dataset():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 1. 加载标定参数
    calibs = load_calib(CONFIG["calib_file"])
    
    # 2. 计算 BEV 尺寸
    bev_w = int((CONFIG["bev_x_range"][1] - CONFIG["bev_x_range"][0]) / CONFIG["bev_resolution"])
    bev_h = int((CONFIG["bev_y_range"][1] - CONFIG["bev_y_range"][0]) / CONFIG["bev_resolution"])
    print(f"BEV Size: {bev_w}x{bev_h} (WxH)")
    
    # 3. 读取 COCO 标注文件
    if not os.path.exists(CONFIG["anno_file"]):
        print(f"Annotation file not found: {CONFIG['anno_file']}")
        return
        
    with open(CONFIG["anno_file"], 'r') as f:
        coco_data = json.load(f)
        
    # 建立 图片名 -> keypoints 的映射
    # 处理 Roboflow 导出的重复数据：一个原始图片可能有多个增强版本 (image_id 不同)
    img_name_to_id = {}
    for img in coco_data['images']:
        name = img['extra']['name'] if 'extra' in img and 'name' in img['extra'] else img['file_name']
        # 只保留第一个遇到的 image_id 以避免重复提取关键点
        if name not in img_name_to_id:
            img_name_to_id[name] = img['id']
            
    valid_ids = set(img_name_to_id.values())
    id_to_name = {v: k for k, v in img_name_to_id.items()}
            
    name_to_kps = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in valid_ids:
            continue
            
        img_name = id_to_name[img_id]
        if 'keypoints' in ann:
            if img_name not in name_to_kps:
                name_to_kps[img_name] = []
            kp_raw = ann['keypoints']
            # 提取点的 (x, y) 忽略 visibility
            for i in range(0, len(kp_raw), 3):
                name_to_kps[img_name].append((kp_raw[i], kp_raw[i+1]))
            
    # 4. 获取图片列表并配对（按时间戳最近邻）
    if not os.path.exists(CONFIG["raw_front_dir"]) or not os.path.exists(CONFIG["raw_rear_dir"]):
        print("Image directories not found.")
        return
        
    front_files = sorted([f for f in os.listdir(CONFIG["raw_front_dir"]) if f.endswith(('.jpg', '.png', '.bmp'))])
    rear_files = sorted([f for f in os.listdir(CONFIG["raw_rear_dir"]) if f.endswith(('.jpg', '.png', '.bmp'))])
    mapping_file = CONFIG.get("pair_mapping_file", "data/pair_mapping.json")
    explicit_pairs = load_explicit_pairs(mapping_file, front_files, rear_files)
    if explicit_pairs is not None and len(explicit_pairs) > 0:
        pairs = explicit_pairs
        print(f"Pairing by explicit mapping file: {mapping_file}, pairs={len(pairs)}")
    else:
        max_gap = CONFIG.get("max_pair_gap", None)
        pairs = pair_by_nearest_timestamp(front_files, rear_files, max_gap=max_gap)
        print(f"Pairing by nearest timestamp with max_gap={max_gap}: {len(pairs)} candidate pairs")
        # Save template for manual correction.
        save_mapping_template(mapping_file, pairs)
        print(f"Saved pairing template: {mapping_file}")
    
    sample_idx = 0
    for front_name, rear_name, ts_gap in pairs:
        front_path = os.path.join(CONFIG["raw_front_dir"], front_name)
        rear_path = os.path.join(CONFIG["raw_rear_dir"], rear_name)
        
        kp_front = name_to_kps.get(front_name, [])
        kp_rear = name_to_kps.get(rear_name, [])
        
        if len(kp_front) != CONFIG["num_keypoints"] or len(kp_rear) != CONFIG["num_keypoints"]:
            print(f"Skipping pair: {front_name} & {rear_name} (gap={ts_gap}, front={len(kp_front)}, rear={len(kp_rear)})")
            continue
            
        # 5. 计算融合的物理坐标并映射到 BEV
        bev_points = []
        for i in range(CONFIG["num_keypoints"]):
            u_f, v_f = kp_front[i]
            u_r, v_r = kp_rear[i]
            
            # 前相机反投影
            x_f, y_f = back_project_to_ground(
                u_f, v_f, 
                calibs['front']['K'], calibs['front']['R'], calibs['front']['T']
            )
            
            # 后相机反投影
            x_r, y_r = back_project_to_ground(
                u_r, v_r, 
                calibs['rear']['K'], calibs['rear']['R'], calibs['rear']['T']
            )
            
            # 取平均
            x_avg = (x_f + x_r) / 2.0
            y_avg = (y_f + y_r) / 2.0
            
            # 转为 BEV 坐标
            bev_u, bev_v = physical_to_bev(
                x_avg, y_avg, 
                CONFIG["bev_x_range"], CONFIG["bev_y_range"], CONFIG["bev_resolution"]
            )
            bev_points.append((bev_u, bev_v))
            
        # 6. 生成热图
        heatmap = generate_gaussian_heatmap((bev_h, bev_w), bev_points, CONFIG["sigma"])
        
        # 7. 保存数据
        sample_dir = os.path.join(CONFIG["output_dir"], f"sample_{sample_idx:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 复制图片
        shutil.copy(front_path, os.path.join(sample_dir, "front.jpg"))
        shutil.copy(rear_path, os.path.join(sample_dir, "rear.jpg"))
        
        # 保存热图
        np.save(os.path.join(sample_dir, "label.npy"), heatmap)
        
        # 保存相机参数
        calib_json = {
            "front": {k: v.tolist() for k, v in calibs["front"].items()},
            "rear": {k: v.tolist() for k, v in calibs["rear"].items()}
        }
        with open(os.path.join(sample_dir, "calib.json"), 'w') as f:
            json.dump(calib_json, f, indent=4)
            
        sample_idx += 1
        
    print(f"Processing complete. Generated {sample_idx} samples in {CONFIG['output_dir']}")

if __name__ == "__main__":
    process_dataset()
