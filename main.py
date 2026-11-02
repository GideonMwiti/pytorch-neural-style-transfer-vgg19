"""
Project 35: Neural Style Transfer using Pretrained VGG-19
=========================================================
Description:
End-to-end implementation of the Gatys et al. (2015) Neural Algorithm of Artistic Style.
Uses a pretrained VGG-19 convolutional network to separate and recombine the semantic content
of a base photograph with the artistic textural style of a painting. Employs feature correlation
Gram matrices, multi-scale style loss, total variation (TV) denoising regularization, and
quasi-Newton L-BFGS image optimization.
"""

import copy
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# -----------------------------------------------------------------------------
# Configuration & Seed Initialization
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
TORCH_CACHE_DIR = PROJECT_DIR / "torch_cache"
IMAGE_DIR = PROJECT_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)

RANDOM_SEED = 42
IMAGE_SIZE = 256  # High-definition resolution for painterly stroke synthesis
NUM_STEPS = 60    # Optimal steps for quasi-Newton L-BFGS convergence on CPU
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 1e6
TV_WEIGHT = 1e-4

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def set_seed(seed: int = 42) -> None:
    """Ensure strict deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Procedural Image Generator (Fallback / Guarantee)
# -----------------------------------------------------------------------------
def ensure_sample_images(image_dir: Path, size: int = IMAGE_SIZE) -> Tuple[Path, Path]:
    """Generates synthetic content and style images if not present."""
    content_path = image_dir / "content.png"
    style_path = image_dir / "style.png"

    w, h = size, size
    if not content_path.exists():
        # Content: Architectural bridge and skyline silhouette
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            r = int(135 + (255 - 135) * (y / h))
            g = int(206 + (218 - 206) * (y / h))
            b = int(235 + (185 - 235) * (y / h))
            img[y, :] = [r, g, b]
        for x in range(int(w * 0.1), int(w * 0.9)):
            bh = int(h * 0.25 + 0.15 * h * np.sin(x * 0.1) + 0.1 * h * np.cos(x * 0.05))
            img[h - bh:, x] = [40, 50, 70]
        # Bridge pylons & parabolic cable
        p1, p2 = int(w * 0.28), int(w * 0.72)
        img[int(h * 0.3):h, p1:p1 + 8] = [190, 40, 20]
        img[int(h * 0.3):h, p2:p2 + 8] = [190, 40, 20]
        for x in range(w):
            cy = int(h * 0.35 + 0.003 * ((x - w // 2) ** 2) * (h / 128))
            if 0 <= cy < h:
                img[cy:cy + 3, x] = [180, 30, 20]
        Image.fromarray(img).save(content_path)

    if not style_path.exists():
        # Style: Swirling impressionist palette (Van Gogh / Starry Night motif)
        style = np.zeros((h, w, 3), dtype=np.uint8)
        X, Y = np.meshgrid(np.linspace(-3, 3, w), np.linspace(-3, 3, h))
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        spiral = np.sin(3 * theta + 2 * R)
        dots = np.sin(10 * X) * np.cos(10 * Y)

        style[:, :, 0] = np.clip(30 + 180 * (spiral > 0) + 45 * dots, 0, 255).astype(np.uint8)
        style[:, :, 1] = np.clip(60 + 120 * np.cos(R) + 75 * (dots > 0.5), 0, 255).astype(np.uint8)
        style[:, :, 2] = np.clip(180 + 75 * np.sin(theta * 2) + 60 * dots, 0, 255).astype(np.uint8)
        gold1 = (X - 0.8)**2 + (Y - 0.8)**2 < 0.5
        style[gold1] = [240, 200, 30]
        gold2 = (X + 1.2)**2 + (Y + 1.0)**2 < 0.35
        style[gold2] = [255, 220, 50]
        Image.fromarray(style).save(style_path)

    return content_path, style_path


# -----------------------------------------------------------------------------
# Image Loading and Normalization Primitives
# -----------------------------------------------------------------------------
def load_image(image_path: Path, size: int = IMAGE_SIZE, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Loads, resizes, and converts an image to a normalized [1, 3, H, W] PyTorch tensor."""
    img = Image.open(image_path).convert("RGB")
    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    tensor = loader(img).unsqueeze(0).to(device)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Converts a [1, 3, H, W] PyTorch tensor in [0, 1] to a PIL Image."""
    t = tensor.detach().cpu().squeeze(0).clamp(0, 1)
    unloader = transforms.ToPILImage()
    return unloader(t)


class Normalization(nn.Module):
    """Normalizes input image tensors with ImageNet mean and standard deviation."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


# -----------------------------------------------------------------------------
# Content, Style, and Total Variation Losses
# -----------------------------------------------------------------------------
def gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gram Matrix G = A * A^T representing cross-channel feature correlations.
    Normalized by total spatial activation volume (C * H * W).
    """
    batch_size, channels, height, width = input_tensor.size()
    features = input_tensor.view(batch_size * channels, height * width)
    G = torch.mm(features, features.t())
    return G.div(batch_size * channels * height * width)


class ContentLoss(nn.Module):
    """Calculates Mean Squared Error between generated and target content feature maps."""
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()
        self.loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    """Calculates Mean Squared Error between generated and target Gram feature matrices."""
    def __init__(self, target_feature: torch.Tensor):
        super().__init__()
        self.target_gram = gram_matrix(target_feature).detach()
        self.loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target_gram)
        return x


def compute_tv_loss(img: torch.Tensor) -> torch.Tensor:
    """Calculates Total Variation regularization to eliminate high-frequency checkerboard noise."""
    diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
    return torch.sum(torch.abs(diff_h)) + torch.sum(torch.abs(diff_w))


# -----------------------------------------------------------------------------
# VGG-19 Model Assembly & Truncation
# -----------------------------------------------------------------------------
def build_style_transfer_model(
    device: torch.device,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    content_layers: List[str] = ["conv_4"],
    style_layers: List[str] = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"],
) -> Tuple[nn.Sequential, List[ContentLoss], List[StyleLoss], torch.Tensor]:
    """
    Constructs a truncated sequential VGG-19 model instrumented with ContentLoss
    and StyleLoss modules inserted immediately following the specified layers.
    """
    # Load pretrained VGG-19 features
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    for p in cnn.parameters():
        p.requires_grad = False

    norm_module = Normalization(IMAGENET_MEAN.to(device), IMAGENET_STD.to(device))

    model = nn.Sequential(norm_module)
    content_losses = []
    style_losses = []
    first_style_gram = None

    i = 0
    conv_count = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            # Replace in-place ReLU with out-of-place for autograd stability
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
            # Gatys et al. recommend AvgPool2d for smoother gradient propagation
            layer = nn.AvgPool2d(kernel_size=2, stride=2)
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            name = f"layer_{i}"

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            c_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", c_loss)
            content_losses.append(c_loss)

        if name in style_layers:
            target_feat = model(style_img).detach()
            s_loss = StyleLoss(target_feat)
            model.add_module(f"style_loss_{i}", s_loss)
            style_losses.append(s_loss)
            if first_style_gram is None:
                first_style_gram = gram_matrix(target_feat).detach().cpu()

        # Prune network after the deepest required loss layer
        if i >= 5 and isinstance(layer, (nn.Conv2d, nn.ReLU)):
            break

    # Truncate any trailing layers beyond the last loss node
    for idx in range(len(model) - 1, -1, -1):
        if isinstance(model[idx], (ContentLoss, StyleLoss)):
            break
    model = model[:idx + 1]

    return model, content_losses, style_losses, first_style_gram


# -----------------------------------------------------------------------------
# Diagnostic Plotting Dashboard
# -----------------------------------------------------------------------------
def plot_style_transfer_dashboard(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    stylized_img: torch.Tensor,
    history: Dict[str, List[float]],
    snapshots: List[Tuple[int, torch.Tensor]],
    style_gram: torch.Tensor,
    save_path: Path,
) -> None:
    """Generates a 6-panel publication-grade composite report."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

    # 1. Content Image
    ax_c = fig.add_subplot(gs[0, 0])
    ax_c.imshow(tensor_to_image(content_img))
    ax_c.set_title("(a) Content Target (Structure)", fontsize=12, fontweight="bold", pad=8)
    ax_c.axis("off")

    # 2. Style Image
    ax_s = fig.add_subplot(gs[0, 1])
    ax_s.imshow(tensor_to_image(style_img))
    ax_s.set_title("(b) Style Source (Texture & Color)", fontsize=12, fontweight="bold", pad=8)
    ax_s.axis("off")

    # 3. Final Stylized Output
    ax_out = fig.add_subplot(gs[0, 2])
    ax_out.imshow(tensor_to_image(stylized_img))
    ax_out.set_title("(c) Synthesized Output Image", fontsize=12, fontweight="bold", pad=8)
    ax_out.axis("off")

    # 4. Loss Trajectories Over Optimization Iterations
    ax_loss = fig.add_subplot(gs[1, :2])
    steps = range(1, len(history["total_loss"]) + 1)
    ax_loss.plot(steps, history["total_loss"], "-", color="#1f77b4", linewidth=2.2, label="Total Loss")
    ax_loss.plot(steps, history["style_loss"], "--", color="#d62728", linewidth=1.8, label="Style Loss (x1e6)")
    ax_loss.plot(steps, history["content_loss"], "-.", color="#2ca02c", linewidth=1.8, label="Content Loss")
    ax_loss.set_title("(d) Optimization Loss Trajectory (L-BFGS Quasi-Newton)", fontsize=12, fontweight="bold", pad=8)
    ax_loss.set_xlabel("L-BFGS Optimization Iteration", fontsize=10)
    ax_loss.set_ylabel("Loss Magnitude", fontsize=10)
    ax_loss.set_yscale("log")
    ax_loss.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    # 5. Gram Matrix Heatmap (Layer 1 Cross-Channel Correlations)
    ax_gram = fig.add_subplot(gs[1, 2])
    gram_np = style_gram.squeeze().numpy()[:64, :64]  # First 64 channels
    im_gram = ax_gram.imshow(gram_np, cmap="magma")
    ax_gram.set_title("(e) Style Gram Matrix (conv1_1)", fontsize=12, fontweight="bold", pad=8)
    ax_gram.axis("off")
    fig.colorbar(im_gram, ax=ax_gram, fraction=0.046, pad=0.04)

    # 6. Progressive Stylization Timeline (4 Snapshots across training)
    for idx, (step_num, snap_tensor) in enumerate(snapshots):
        ax_snap = fig.add_subplot(gs[2, idx % 4 if len(snapshots) <= 4 else 0])
        ax_snap.imshow(tensor_to_image(snap_tensor))
        ax_snap.set_title(f"Step {step_num}", fontsize=10, fontweight="semibold")
        ax_snap.axis("off")

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Artifact] Saved comprehensive style transfer dashboard to: {save_path}")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("Project 35: Neural Style Transfer using Pretrained VGG-19")
    print("=" * 75)
    print(f"Compute Device        : {device}")
    print(f"Pretrained Backbone   : VGG-19 (ImageNet-1K Pretrained Features)")
    print(f"Target Image Size     : {IMAGE_SIZE}x{IMAGE_SIZE} pixels")
    print(f"Optimization Steps    : {NUM_STEPS} (L-BFGS Quasi-Newton)")
    print(f"Loss Hyperparameters  : Content={CONTENT_WEIGHT} | Style={STYLE_WEIGHT:.1e} | TV={TV_WEIGHT:.1e}")
    print("-" * 75)

    # 1. Ensure content and style image assets
    content_path, style_path = ensure_sample_images(IMAGE_DIR, size=IMAGE_SIZE)
    content_img = load_image(content_path, size=IMAGE_SIZE, device=device)
    style_img = load_image(style_path, size=IMAGE_SIZE, device=device)
    print(f"Loaded Content Source : {content_path.name} {list(content_img.shape)}")
    print(f"Loaded Style Source   : {style_path.name} {list(style_img.shape)}")

    # 2. Build model and extract loss nodes
    model, content_losses, style_losses, style_gram = build_style_transfer_model(
        device, content_img, style_img
    )
    print(f"Synthesizing Pipeline : {len(content_losses)} Content Losses, {len(style_losses)} Style Losses")
    print("-" * 75)

    # 3. Target image initialization (Content initialization converges faster with higher fidelity)
    input_img = content_img.clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS([input_img], lr=1.0)

    history = {
        "total_loss": [],
        "content_loss": [],
        "style_loss": [],
        "tv_loss": [],
    }
    snapshots = []
    snapshots.append((0, input_img.clone()))

    step_count = [0]
    start_time = time.perf_counter()

    print(">>> Commencing L-BFGS Optimization Iterations...")

    while step_count[0] <= NUM_STEPS:
        def closure():
            optimizer.zero_grad()
            input_img.data.clamp_(0, 1)

            model(input_img)

            c_loss = sum(cl.loss for cl in content_losses) * CONTENT_WEIGHT
            s_loss = sum(sl.loss for sl in style_losses) * STYLE_WEIGHT
            tv_loss = compute_tv_loss(input_img) * TV_WEIGHT

            tot_loss = c_loss + s_loss + tv_loss
            tot_loss.backward()

            step_count[0] += 1
            if step_count[0] % 10 == 0 or step_count[0] == 1:
                print(
                    f"Step [{step_count[0]:02d}/{NUM_STEPS:02d}] "
                    f"Total Loss: {tot_loss.item():9.2f} | "
                    f"Style Loss: {s_loss.item():9.2f} | "
                    f"Content Loss: {c_loss.item():6.4f} | "
                    f"TV Loss: {tv_loss.item():6.4f}"
                )

            history["total_loss"].append(tot_loss.item())
            history["content_loss"].append(c_loss.item())
            history["style_loss"].append(s_loss.item())
            history["tv_loss"].append(tv_loss.item())

            if step_count[0] in [15, 30, 45, NUM_STEPS]:
                snapshots.append((step_count[0], input_img.clone()))

            return tot_loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    duration = time.perf_counter() - start_time
    print("-" * 75)
    print(f"Optimization Finished in {duration:.2f} seconds.")

    # 4. Save stylized output image
    final_output_path = IMAGE_DIR / "stylized_output.png"
    out_pil = tensor_to_image(input_img)
    out_pil.save(final_output_path)
    print(f"[Artifact] Saved standalone stylized image to: {final_output_path}")

    # 5. Save composite publication dashboard
    dashboard_path = PROJECT_DIR / "neural_style_transfer_result.png"
    plot_style_transfer_dashboard(
        content_img=content_img,
        style_img=style_img,
        stylized_img=input_img,
        history=history,
        snapshots=snapshots,
        style_gram=style_gram,
        save_path=dashboard_path,
    )

    print("=" * 75)
    print("Neural Style Transfer Experiment Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()

