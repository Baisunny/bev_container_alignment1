# Data Notice (Public Version)

This repository is a desensitized public version.

Removed from version control:
- real processed samples under `data/processed_before_maxgap_20260507/`
- private annotation file `data/annotations/_annotations.coco.json`
- private mapping file `data/pair_mapping.json`

How to use:
1. Put your own raw images and annotations under `data/raw/` and `data/annotations/`.
2. Run `python3 scripts/generate_bev_labels.py` to build processed samples.
