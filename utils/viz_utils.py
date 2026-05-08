import cv2
import numpy as np
import torch
from configs.config import Config

def visualize_bev(heatmap, points_mm=None, spreader_centroid=None, container_centroid=None):
    """
    Visualize BEV heatmap and detected points.
    heatmap: (8, H, W)
    points_mm: (8, 2)
    spreader_centroid: (2,)
    container_centroid: (2,)
    """
    h, w = Config.bev_h, Config.bev_w
    
    # Combined heatmap (max across classes)
    combined_hm = torch.max(heatmap, dim=0)[0].detach().cpu().numpy()
    combined_hm = (combined_hm * 255).astype(np.uint8)
    combined_hm = cv2.applyColorMap(combined_hm, cv2.COLORMAP_JET)
    
    # Scale to larger size for visibility
    vis = cv2.resize(combined_hm, (512, 512))
    scale_x = 512 / w
    scale_y = 512 / h
    
    # Draw points if provided
    if spreader_centroid is not None:
        # Convert mm to grid to vis
        u = (spreader_centroid[0] - Config.x_range[0]) / Config.bev_resolution * scale_x
        v = (spreader_centroid[1] - Config.y_range[0]) / Config.bev_resolution * scale_y
        cv2.circle(vis, (int(u), int(v)), 5, (0, 0, 255), -1) # Red for spreader
        cv2.putText(vis, "Spreader", (int(u), int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    if container_centroid is not None:
        u = (container_centroid[0] - Config.x_range[0]) / Config.bev_resolution * scale_x
        v = (container_centroid[1] - Config.y_range[0]) / Config.bev_resolution * scale_y
        cv2.circle(vis, (int(u), int(v)), 5, (0, 255, 0), -1) # Green for container
        cv2.putText(vis, "Container", (int(u), int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    return vis

def draw_offsets(img_f, img_r, dx, dy):
    """
    Overlay offset text on images.
    """
    combined = np.hstack([img_f, img_r])
    h, w, _ = combined.shape
    cv2.putText(combined, f"DX: {dx:.2f}mm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, f"DY: {dy:.2f}mm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return combined
