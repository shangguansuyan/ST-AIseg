# ST-AIseg
AI-based segmentation in spatial transcriptomics

!(amns.jpg)

## Overview

This script processes Hematoxylin and Eosin (H&E) stained images from spatial transcriptomics data (e.g., from SpaceRanger output) to perform histology-based spatial clustering. It mimics aspects of SpaGCN by integrating deep learning features from EfficientNet, nuclear segmentation via StarDist, spatial coordinates, and graph-based clustering using Louvain communities. The goal is to identify spatial domains or clusters based on histological and positional information.

Key steps:
- Load H&E image and spot positions.
- Extract image patches for each spot.
- Use StarDist for nuclear segmentation on the entire image.
- Extract features using EfficientNet (pre-trained on ImageNet) and nuclear properties.
- Combine features with normalized spatial coordinates.
- Build a graph with spatial neighbors and feature similarities.
- Perform Louvain clustering.
- Visualize clusters as tiled squares on the image coordinates.

This is designed for datasets like Visium spatial transcriptomics, but can be adapted.

## Requirements

### Python Version
- Python 3.9 (or compatible; tested on 3.9)

### Dependencies
The script relies on the following libraries. Install them via pip:

- **torch** and **torchvision**: For EfficientNet model and data loading.
- **numpy**, **pandas**: Data manipulation.
- **PIL (Pillow)**: Image handling.
- **scikit-learn**: PCA, cosine similarity, KDTree.
- **scipy**: Spatial tools.
- **networkx**: Graph construction and Louvain clustering.
- **matplotlib**: Visualization.
- **stardist** and **csbdeep**: Nuclear segmentation (pre-trained model for H&E).
- **scikit-image (skimage)**: Region properties for nuclear features.

Note: The script uses a pre-trained StarDist model (`2D_versatile_he`) which is downloaded automatically on first run. EfficientNet uses pre-trained weights from torchvision.

Hardware: GPU recommended for EfficientNet feature extraction (falls back to CPU if unavailable). Large images may require significant RAM; tiling is used for StarDist to handle this.

## Input Data

- **Image Path**: H&E image (e.g., `spatial/oi.png` from SpaceRanger).
- **Positions File**: CSV with spot positions (e.g., `spatial/tissue_positions_list.csv`).
- Data is assumed to be in a directory like `/mnt/d/.../Report/1.SpaceRanger/WT_3/`.

The script filters for in-tissue spots and uses pixel coordinates for patch extraction.

## Usage

1. **Set Paths**: Update `data_path` and `image_path` in the script to point to your data.
2. **Run the Script**:
   ```
   python HE_ST.py
   ```
   - This will process the image, extract features, perform clustering, and display a visualization.
   - Outputs:
     - Patches saved in `patches/` directory (one PNG per spot, named by barcode).
     - StarDist mask saved as `stardist_mask.png`.
     - Clustering results added to the `positions` DataFrame (column: `cluster`).
     - Matplotlib plot showing clustered spots as colored squares.

3. **Customization**:
   - **Patch Size**: Dynamically computed from median nearest-neighbor distance; adjust if needed.
   - **PCA Components**: Set to 50; tune via `n_components`.
   - **Spatial Weight**: `spatial_weight = 0.1`; balances histology vs. spatial info (like SpaGCN's alpha).
   - **Neighbors**: `k_neighbors = 10`; number of spatial neighbors for graph.
   - **StarDist Thresholds**: `prob_thresh=0.5`, `nms_thresh=0.4`; adjust for better segmentation.
   - **Tiling**: `target_tile_size = 1024`; reduce for low-memory systems.

## Output Explanation

- **Features**:
  - EfficientNet: 1280-dimensional features per patch, reduced to 50 via PCA.
  - Nuclear: Per-spot stats (count, mean/std of area, perimeter, eccentricity, intensity, solidity) from StarDist labels.
  - Combined with weighted spatial coordinates.

- **Graph**:
  - Nodes: Spots.
  - Edges: Between k-nearest spatial neighbors, weighted by cosine similarity of combined features.

- **Clustering**:
  - Louvain algorithm on the weighted graph.
  - Labels assigned to spots (0-based cluster IDs).

- **Visualization**:
  - Tiled squares (size: patch_size) colored by cluster.
  - Uses HSV colormap for distinct colors.

## Limitations and Notes

- **Memory**: Large images (>10k x 10k pixels) may require tiling adjustments or more RAM.
- **No Gene Expression**: This is histology-only; for full SpaGCN-like analysis, integrate with expression data.
- **CPU Mode for StarDist**: Forced to CPU; enable GPU in StarDist config if available.
- **PIL Truncation**: Enabled to handle potentially corrupted images.
- **No Multiprocessing**: Dataloader uses `num_workers=0` to avoid issues; enable if stable.
- **Visualization**: Inverted y-axis to match spatial conventions.

For issues or extensions, refer to the source libraries' documentation (e.g., StarDist, EfficientNet).

## License

This script is provided as-is for research purposes. No license specified; adapt freely.
