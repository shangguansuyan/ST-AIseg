import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import torch
from torchvision import transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import KDTree
import networkx as nx
from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops_table, label
# Fix for broken data stream in PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Set paths
data_path = "/mnt/d/研究/盛书颜/自测空转/Report/1.SpaceRanger/WT_3/"
image_path = os.path.join(data_path, "spatial/oi.png")
# Allow large image loading
Image.MAX_IMAGE_PIXELS = None
# Load the image as PIL and NumPy for StarDist
image = Image.open(image_path)
image_np = np.array(image) # Convert to NumPy array for StarDist
image_height, image_width = image_np.shape[:2] # Use np shape for accuracy
print(f"Image size: {image_width}x{image_height}")
# Load spatial positions from CSV as specified
positions_file = os.path.join(data_path, "spatial/tissue_positions_list.csv")
positions = pd.read_csv(positions_file, header=None, names=['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres'])
# Filter for spots in tissue
positions = positions[positions['in_tissue'] == 1]
# Coordinates and barcodes
coords = positions[['pxl_row_in_fullres', 'pxl_col_in_fullres']].values
barcodes = positions['barcode'].values
# Compute dynamic patch size based on median nearest neighbor distance
tree = KDTree(coords)
dists, _ = tree.query(coords, k=2)
min_dists = dists[:, 1]
patch_size = int(np.median(min_dists))
print(f"Computed patch_size: {patch_size}")
half_patch = patch_size // 2
# Create directory for patches
patches_dir = os.path.join(data_path, "patches")
os.makedirs(patches_dir, exist_ok=True)
# Function to extract and save patch
def extract_and_save_patch(img, center_x, center_y, barcode):
    left = max(0, center_y - half_patch)
    top = max(0, center_x - half_patch)
    right = min(img.width, center_y + half_patch)
    bottom = min(img.height, center_x + half_patch)
    patch = img.crop((left, top, right, bottom))
    # Pad if necessary
    if patch.size != (patch_size, patch_size):
        padded = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
        pad_left = half_patch - (right - left) // 2
        pad_top = half_patch - (bottom - top) // 2
        padded.paste(patch, (pad_left, pad_top))
        patch = padded
    # Save patch with barcode name
    patch_path = os.path.join(patches_dir, f"{barcode}.png")
    patch.save(patch_path)
    return patch
# Create dataset for patches (only for EfficientNet, since StarDist on whole image)
class PatchDataset(Dataset):
    def __init__(self, image, coords, barcodes, transform=None):
        self.image = image
        self.coords = coords
        self.barcodes = barcodes
        self.transform = transform
    def __len__(self):
        return len(self.coords)
    def __getitem__(self, idx):
        x, y = self.coords[idx]
        barcode = self.barcodes[idx]
        patch = extract_and_save_patch(self.image, int(x), int(y), barcode)
        if self.transform:
            patch = self.transform(patch)
        return patch
# Preprocess transform for EfficientNet
preprocess = transforms.Compose([
    transforms.Resize(480), # EfficientNet_v2_m input size
    transforms.CenterCrop(480),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Create dataset and dataloader (set num_workers=0 to avoid multiprocessing issues)
dataset = PatchDataset(image, coords, barcodes, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
# Load EfficientNet model
weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
model = efficientnet_v2_m(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1]) # Remove classifier for feature extraction
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Load StarDist model (pretrained on H&E for nuclei, using CPU)
print("Loading StarDist model...")
stardist_model = StarDist2D.from_pretrained('2D_versatile_he') # Correct model for H&E; adjust if needed
stardist_model.config.use_gpu = False # Force CPU mode
# Run StarDist on the whole image with tiling to handle large size
print("Running StarDist on the entire image with tiling...")
target_tile_size = 1024 # Adjust based on memory; smaller for less RAM
n_tiles_y = (image_height + target_tile_size - 1) // target_tile_size
n_tiles_x = (image_width + target_tile_size - 1) // target_tile_size
n_tiles = (n_tiles_y, n_tiles_x, 1) # Add 1 for channels since image is 3D (H, W, C)
print(f"Using n_tiles: {n_tiles}")
image_norm = normalize(image_np) # Normalize image for StarDist
labels, _ = stardist_model.predict_instances(image_norm, prob_thresh=0.5, nms_thresh=0.4, n_tiles=n_tiles) # Adjust thresholds if needed
# Save the StarDist mask as an image
mask_path = os.path.join(data_path, "stardist_mask.png")
plt.imsave(mask_path, labels, cmap='gray') # Save as grayscale; adjust cmap if needed
print(f"StarDist mask saved to: {mask_path}")
# Intensity image for the whole image (grayscale)
intensity_img = np.mean(image_np, axis=-1)
# Extract features: EfficientNet + StarDist nuclear textures per spot
features = []
with torch.no_grad():
    for batch in dataloader:
        # EfficientNet features
        batch = batch.to(device)
        feat = model(batch).cpu().numpy()
        features.append(feat)
features = np.concatenate(features, axis=0)
features = features.reshape(features.shape[0], -1) # Flatten
# Now extract nuclear features per spot using the global labels
nuclear_features_list = []
for idx in range(len(coords)):
    x, y = coords[idx]
    # Define spot region bounds
    left = max(0, int(y) - half_patch)
    top = max(0, int(x) - half_patch)
    right = min(image_width, int(y) + half_patch)
    bottom = min(image_height, int(x) + half_patch)
   
    # Extract sub-label and sub-intensity for the spot region
    sub_label = labels[top:bottom, left:right]
    sub_intensity = intensity_img[top:bottom, left:right]
   
    # Relabel sub_label to ensure unique labels in region
    sub_label_relabeled = label(sub_label > 0) # Binarize and relabel
   
    if np.any(sub_label_relabeled):
        props = regionprops_table(sub_label_relabeled, intensity_image=sub_intensity,
                                  properties=['area', 'perimeter', 'eccentricity', 'mean_intensity', 'solidity'])
        # Aggregate: mean and std of properties, plus count
        nuclear_feat = [
            len(props['area']), # Nuclei count
            np.mean(props['area']), np.std(props['area']),
            np.mean(props['perimeter']), np.std(props['perimeter']),
            np.mean(props['eccentricity']), np.std(props['eccentricity']),
            np.mean(props['mean_intensity']), np.std(props['mean_intensity']),
            np.mean(props['solidity']), np.std(props['solidity'])
        ]
    else:
        nuclear_feat = [0] * 11 # Zero if no nuclei
    nuclear_features_list.append(nuclear_feat)
nuclear_features = np.array(nuclear_features_list) # Shape: (n_spots, 11)
print(f"Extracted EfficientNet features shape: {features.shape}")
print(f"Extracted nuclear features shape: {nuclear_features.shape}")
# Reduce dimensions with PCA on EfficientNet features
pca = PCA(n_components=50)
reduced_features = pca.fit_transform(features)
# Combine with nuclear features
extended_features = np.hstack((reduced_features, nuclear_features))
# Normalize coordinates for inclusion (imitating SpaGCN's spatial integration)
norm_coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-6)
# Combine features with spatial coordinates (weight spatial features as in SpaGCN, e.g., alpha=0.1 for spatial)
spatial_weight = 0.1 # Similar to SpaGCN's alpha parameter for balancing histology and expression (here features)
combined_features = np.hstack((extended_features, spatial_weight * norm_coords))
# Build graph imitating SpaGCN: spatial neighbors with weights from feature + spatial similarity
G = nx.Graph()
# Add nodes
for i in range(len(positions)):
    G.add_node(i)
# Use KDTree for spatial neighbors (SpaGCN uses radius or kNN for adjacency)
k_neighbors = 10 # Similar to SpaGCN's neighbor selection
_, indices = tree.query(coords, k=k_neighbors)
# Compute adjacency weights: in SpaGCN, weights can be based on expression similarity + histology
# Here, use cosine on combined_features for similarity
sim_matrix = cosine_similarity(combined_features)
# Add edges: only for spatial neighbors, weight = similarity (mimicking SpaGCN's weighted graph)
for i in range(len(indices)):
    for j in indices[i][1:]: # Skip self
        weight = sim_matrix[i, j] # Use similarity as weight, no threshold to mimic full graph
        if weight > 0: # Add all positive weights
            G.add_edge(i, j, weight=weight)
# Perform Louvain clustering (SpaGCN uses Leiden/Louvain for clustering on graph)
partition = louvain_communities(G, weight='weight')
labels = np.zeros(len(positions), dtype=int)
for cluster_id, nodes in enumerate(partition):
    for node in nodes:
        labels[node] = cluster_id
# Assign labels back to positions
positions['cluster'] = labels
# Visualize with small squares tiled (instead of points), using default Python colormap (tab10 for discrete clusters)
fig, ax = plt.subplots(figsize=(10, 10))
num_clusters = len(np.unique(labels))
colors = plt.cm.hsv(np.linspace(0, 1, num_clusters))
cmap = ListedColormap(colors)
for idx, (x, y, cluster) in enumerate(zip(positions['pxl_row_in_fullres'], positions['pxl_col_in_fullres'], positions['cluster'])):
    rect = Rectangle((y - half_patch, x - half_patch), patch_size, patch_size,
                     facecolor=cmap(cluster), edgecolor='none', alpha=0.8)
    ax.add_patch(rect)
# Set limits to match image dimensions
ax.set_xlim(0, image_width)
ax.set_ylim(image_height, 0) # Invert y-axis for spatial plots
ax.set_aspect('equal')
ax.set_title("Spatial Clustering with Louvain (Tiled Squares)")
plt.show()