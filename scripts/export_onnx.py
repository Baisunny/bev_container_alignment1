import torch
import onnx
from models.detector import BEVFormerDetector
from configs.config import Config

def export():
    # Model
    model = BEVFormerDetector(embed_dim=Config.bev_dim, num_layers=Config.num_bev_layers)
    model.eval()
    
    # Dummy input
    dummy_img = torch.randn(1, 2, 3, Config.img_h, Config.img_w)
    dummy_calibs = torch.randn(1, 2, 3, 4)
    
    # Export
    torch.onnx.export(
        model,
        (dummy_img, dummy_calibs),
        "bevformer.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['img', 'calibs'],
        output_names=['heatmap', 'offset'],
        dynamic_axes={
            'img': {0: 'batch_size'},
            'calibs': {0: 'batch_size'},
            'heatmap': {0: 'batch_size'},
            'offset': {0: 'batch_size'}
        }
    )
    print("Exported to bevformer.onnx")

if __name__ == '__main__':
    export()
