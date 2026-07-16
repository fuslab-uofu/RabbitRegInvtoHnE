#Import libs->
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torchstain


#paths to Hne data of interest->
all_slides = {
    "R23-055": "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R23-055/HnE/Block07/HnE_IMG_0034.tif",
    "R24-082": "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R24-082/HnE/Block05/HnE_IMG_0030.tif",
    "R24-240": "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R24-240/HnE/Block03/HnE_IMG_0047.tif",
    "R24-103": "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R24-103/HnE/Block06/HnE_IMG_0032.tif",
    "R24-101": "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R24-101/HnE/Block11/HnE_IMG_0032.tif",
    "R24-058": "/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/R24-058/HnE/Block06/HnE_IMG_0038.tif",
}

#tweak this to try a different rabbit as the normalization target
REFERENCE_NAME = "R23-055"


def load_rgb(path):
    img = tifffile.imread(path)
    return img[..., :3]  # drop alpha channel if present


def show_slides(images, titles, suptitle):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


reference_img = load_rgb(all_slides[REFERENCE_NAME])
test_slides = {name: path for name, path in all_slides.items() if name != REFERENCE_NAME}
test_imgs = {name: load_rgb(path) for name, path in test_slides.items()}

#Visulization of slides side by side
show_slides(
    [reference_img] + list(test_imgs.values()),
    [f"{REFERENCE_NAME} (reference)"] + list(test_imgs.keys()),
    "Before Macenko normalization",
)


#Stain normalization
normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
normalizer.fit(reference_img)

normalized_imgs = {}
stain_channels = {}  # name -> (H, E) concentration maps from torchstain
for name, img in test_imgs.items():
    norm_img, H, E = normalizer.normalize(I=img, stains=True)
    normalized_imgs[name] = np.clip(norm_img, 0, 255).astype(np.uint8)
    stain_channels[name] = (H, E)


#visualization of slides side by side post normalization
show_slides(
    [reference_img] + list(normalized_imgs.values()),
    [f"{REFERENCE_NAME} (reference)"] + list(normalized_imgs.keys()),
    "After Macenko normalization",
)


#--- Math visualization: OD color cloud + estimated stain vectors ---
# This reimplements just the stain-vector-estimation steps of Macenko (RGB -> optical
# density -> PCA -> extreme angles) purely for visualization. The actual normalization
# above is still done by torchstain; this is a separate, self-contained explainer.
def estimate_stain_vectors(img, od_threshold=0.15, angular_percentile=1):
    pixels = img.reshape(-1, 3).astype(np.float64)
    od = -np.log10((pixels + 1) / 256)  # RGB -> optical density

    tissue_mask = (od > od_threshold).any(axis=1)  # drop background/white pixels
    od_tissue = od[tissue_mask]

    cov = np.cov(od_tissue.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = eigvecs[:, np.argsort(eigvals)[-2:]]  # 3x2: top 2 principal directions

    proj = od_tissue @ top2  # Nx2 projection onto that plane
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    min_angle = np.percentile(angles, angular_percentile)
    max_angle = np.percentile(angles, 100 - angular_percentile)

    v_min = np.array([np.cos(min_angle), np.sin(min_angle)])
    v_max = np.array([np.cos(max_angle), np.sin(max_angle)])
    stain_od_1 = top2 @ v_min  # extreme stain direction, back in RGB-OD space
    stain_od_2 = top2 @ v_max

    return {
        "proj": proj,
        "v_min": v_min,
        "v_max": v_max,
        "stain_od_1": stain_od_1,
        "stain_od_2": stain_od_2,
    }


def od_to_rgb_swatch(stain_od):
    # crude inverse of the OD transform, just to get a representative color swatch
    rgb = 256 * (10 ** -stain_od) - 1
    return np.clip(rgb, 0, 255).astype(np.uint8)


def plot_stain_vector_math(img, title, n_sample=20000, od_threshold=0.15, angular_percentile=1):
    result = estimate_stain_vectors(img, od_threshold, angular_percentile)
    proj = result["proj"]
    if proj.shape[0] > n_sample:
        idx = np.random.choice(proj.shape[0], n_sample, replace=False)
        proj = proj[idx]

    arrow_scale = np.percentile(np.linalg.norm(proj, axis=1), 95)

    fig, (ax_scatter, ax_swatch) = plt.subplots(1, 2, figsize=(9, 4.5), gridspec_kw={"width_ratios": [3, 1]})

    ax_scatter.scatter(proj[:, 0], proj[:, 1], s=1, alpha=0.15, color="gray")
    for v, color in [(result["v_min"], "tab:blue"), (result["v_max"], "tab:red")]:
        ax_scatter.annotate(
            "", xy=(v[0] * arrow_scale, v[1] * arrow_scale), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, linewidth=2),
        )
    ax_scatter.set_title(f"OD color cloud + stain directions\n{title}", fontsize=9)
    ax_scatter.set_xlabel("PC1 (optical density space)")
    ax_scatter.set_ylabel("PC2 (optical density space)")
    ax_scatter.axis("equal")

    for i, stain_key in enumerate(["stain_od_1", "stain_od_2"]):
        swatch = od_to_rgb_swatch(result[stain_key]).reshape(1, 1, 3)
        ax_swatch.imshow(np.tile(swatch, (1, 1, 1)), extent=(0, 1, 1 - i, 2 - i))
    ax_swatch.set_xlim(0, 1)
    ax_swatch.set_ylim(0, 2)
    ax_swatch.set_xticks([])
    ax_swatch.set_yticks([0.5, 1.5])
    ax_swatch.set_yticklabels(["stain 2 (red arrow)", "stain 1 (blue arrow)"], fontsize=8)
    ax_swatch.set_title("approx. stain color", fontsize=9)

    plt.tight_layout()
    plt.show()


for name, img in {REFERENCE_NAME: reference_img, **test_imgs}.items():
    plot_stain_vector_math(img, name)


#--- Hematoxylin / Eosin channel breakdown (from torchstain's own normalize() output) ---
def show_stain_channels(names, suptitle):
    fig, axes = plt.subplots(3, len(names), figsize=(4 * len(names), 10))
    if len(names) == 1:
        axes = axes.reshape(3, 1)
    for i, name in enumerate(names):
        H, E = stain_channels[name]
        axes[0, i].imshow(normalized_imgs[name])
        axes[0, i].set_title(f"{name}\nnormalized", fontsize=9)
        axes[1, i].imshow(H, cmap="gray")
        axes[1, i].set_title("hematoxylin channel", fontsize=9)
        axes[2, i].imshow(E, cmap="gray")
        axes[2, i].set_title("eosin channel", fontsize=9)
        for row in range(3):
            axes[row, i].axis("off")
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


show_stain_channels(list(test_imgs.keys()), "Separated stain channels (post-normalization)")
