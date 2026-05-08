import numpy as np

class Config:
    # --- BEV Settings ---
    # BEV grid range in mm
    x_range = [-500, 500]  # Left to Right
    y_range = [-500, 500]  # Front to Back
    bev_resolution = 4.0   # mm/pixel
    bev_h = int((y_range[1] - y_range[0]) / bev_resolution)
    bev_w = int((x_range[1] - x_range[0]) / bev_resolution)
    
    # --- Image Settings ---
    img_h = 480
    img_w = 640
    input_shape = (480, 640)
    
    # --- Model Settings ---
    num_bev_layers = 3
    bev_dim = 256
    num_points_in_pillar = 4
    
    # --- Training Settings ---
    batch_size = 8
    lr = 2e-4
    epochs = 100
    weight_decay = 0.01
    
    # --- Classes ---
    # 8 points total: 4 for spreader, 4 for container
    num_classes = 8
    
    # --- Camera Parameters ---
    # (These will be loaded from yaml in the dataset but stored here for reference)
    # Usually we define the BEV plane at Z=0 in a "world" coordinate system.
    # The cameras are positioned relative to this world origin.
