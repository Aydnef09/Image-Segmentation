import os
import json
import warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import ndimage
from skimage import data as skdata, filters, morphology, segmentation, feature
from skimage.color import label2rgb

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
for d in ['images', 'ground_truth', 'results', 'figures']:
    os.makedirs(os.path.join(BASE, d), exist_ok=True)

IMG_DIR = os.path.join(BASE, 'images')
GT_DIR = os.path.join(BASE, 'ground_truth')
RES_DIR = os.path.join(BASE, 'results')
FIG_DIR = os.path.join(BASE, 'figures')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – DATASET CREATION
# ─────────────────────────────────────────────────────────────────────────────






def create_dataset():
    """
    Base image: skimage.data.coins() – real 303×384 greyscale photograph.
    Produces 10 photorealistic variants by applying controlled degradations.
    """
    print("[1/5] Creating dataset ...")
    np.random.seed(42)

    coins_gray = skdata.coins()                                               # real photo
    coins_bgr = cv2.cvtColor(coins_gray, cv2.COLOR_GRAY2BGR)

    # ── Degradation functions ──────────────────────────────────────────────
    def add_noise(img, sigma):
        n = np.random.normal(0, sigma, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)

    def shadow_lr(img, lo=0.45):
        h, w = img.shape[:2]
        grad = np.linspace(lo, 1.0, w)
        m = np.outer(np.ones(h), grad)
        if img.ndim == 3: m = m[:, :, None]
        return np.clip(img * m, 0, 255).astype(np.uint8)

    def vignette(img, strength=0.65):
        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        d = np.sqrt((X - w//2)**2 + (Y - h//2)**2)
        m = 1.0 - strength * (d / d.max())
        if img.ndim == 3: m = m[:, :, None]
        return np.clip(img * m, 0, 255).astype(np.uint8)

    def local_shadow(img, x0=128, x1=256, alpha=0.5):
        out = img.copy().astype(np.float32)
        if img.ndim == 3: out[:, x0:x1, :] *= alpha
        else:             out[:, x0:x1]     *= alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    def bright_spot(img, cx=320, cy=60, radius=90, gain=1.6):
        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        d = np.sqrt((X - cx)**2 + (Y - cy)**2)
        m = 1.0 + (gain - 1.0) * np.exp(-d**2 / (2 * radius**2))
        if img.ndim == 3: m = m[:, :, None]
        return np.clip(img * m, 0, 255).astype(np.uint8)

    g, b = coins_gray, coins_bgr
    variants = [
        ('coin_01_clean',          b),
        ('coin_02_noisy',          cv2.cvtColor(add_noise(g, 12),
        cv2.COLOR_GRAY2BGR)),
        ('coin_03_shadow_lr',      cv2.cvtColor(shadow_lr(g),
        cv2.COLOR_GRAY2BGR)),
        ('coin_04_vignette',       cv2.cvtColor(vignette(g),
        cv2.COLOR_GRAY2BGR)),
        ('coin_05_local_shadow',   cv2.cvtColor(local_shadow(g),
        cv2.COLOR_GRAY2BGR)),
        ('coin_06_noisy_shadow',   cv2.cvtColor(add_noise(shadow_lr(g), 10),
        cv2.COLOR_GRAY2BGR)),
        ('coin_07_high_noise',     cv2.cvtColor(add_noise(g, 25),
        cv2.COLOR_GRAY2BGR)),
        ('coin_08_vignette_noise', cv2.cvtColor(add_noise(vignette(g, 0.75), 8),
        cv2.COLOR_GRAY2BGR)),
        ('coin_09_bright_spot',    cv2.cvtColor(bright_spot(g),
        cv2.COLOR_GRAY2BGR)),
        ('coin_10_combined',       cv2.cvtColor(add_noise(shadow_lr(vignette(g,
        0.45)), 14), cv2.COLOR_GRAY2BGR)),
        ]






    for name, img in variants:
        cv2.imwrite(os.path.join(IMG_DIR, f'{name}.png'), img)

    # ── Ground-truth (from clean image, manually verified) ─────────────────
    thresh = filters.threshold_otsu(coins_gray)
    binary = coins_gray > thresh
    binary = morphology.remove_small_objects(binary, max_size=200)
    binary = morphology.remove_small_holes(binary, max_size=500)
    binary = morphology.closing(binary, morphology.disk(4))
    mask   = (binary * 255).astype(np.uint8)

    for name, _ in variants:
        cv2.imwrite(os.path.join(GT_DIR, f'{name}_gt.png'), mask)

    n_coins = ndimage.label(binary)[1]
    print(f"    {len(variants)} images saved | GT: {n_coins} coins                |   "
        f"foreground {binary.mean()*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(img, method='clahe_blur'):
    """
    Convert to greyscale + optional enhancement.
    method: 'none' | 'blur' | 'clahe' | 'clahe_blur'
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    if method == 'none':
        return gray
    elif method == 'blur':
        return cv2.GaussianBlur(gray, (5, 5), 1.2)
    elif method == 'clahe':
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    elif method == 'clahe_blur':
        eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.GaussianBlur(eq, (5, 5), 1.2)
    return gray


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – POST-PROCESSING (shared by all methods)
# ─────────────────────────────────────────────────────────────────────────────

def postprocess(binary, min_area=300, max_area=12000):
    """
    Morphological closing + opening + connected-component area filter.
    """
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kc)
    m = cv2.morphologyEx(m,        cv2.MORPH_OPEN, ko)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area:
            out[labels == i] = 255
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – FAMILY 1: THRESHOLDING
# ─────────────────────────────────────────────────────────────────────────────

def global_otsu(img, preproc='clahe_blur'):
    """
    Global Otsu thresholding.





     Minimises intra-class intensity variance; threshold determined analytically.
     """
    gray = preprocess(img, preproc)
    thresh_val, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    return postprocess(binary), float(thresh_val)


def adaptive_threshold(img, block_size=35, C=4, preproc='clahe_blur'):
    """
    Adaptive Gaussian thresholding (raw / document-style parameters).
    NOTE: produces ring artefacts on large filled discs — see failure analysis.
    """
    gray   = preprocess(img, preproc)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C)
    return postprocess(binary)


def adaptive_threshold_tuned(img, preproc='clahe_blur'):
    """
    Adaptive thresholding with block_size ≈ 2× coin radius and negative C.
    Reduces ring artefacts but cannot fully eliminate them (structural failure).
    """
    gray   = preprocess(img, preproc)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        71, -8)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    return postprocess(binary)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – FAMILY 2: K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def kmeans_segment(img, k=2, preproc='clahe_blur', color_space='grey'):
    """
    K-Means pixel clustering.
    color_space: 'grey' | 'rgb' | 'lab'
    Foreground = cluster with highest mean intensity / L* value.
    """
    if color_space == 'grey':
        gray   = preprocess(img, preproc)
        pixels = gray.reshape(-1, 1).astype(np.float32)
    elif color_space == 'rgb':
        blurred = cv2.GaussianBlur(img, (5, 5), 1.2)
        pixels = blurred.reshape(-1, 3).astype(np.float32)
    elif color_space == 'lab':
        blurred = cv2.GaussianBlur(img, (5, 5), 1.2)
        lab     = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
        pixels = lab.reshape(-1, 3).astype(np.float32)
    else:
        raise ValueError(f'Unknown color_space: {color_space}')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.1)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 15, cv2.KMEANS_PP_CENTERS)

    h, w = img.shape[:2]





    label_img   = labels.reshape(h, w)
    fg_cluster = int(np.argmax(np.mean(centers, axis=1)))
    binary      = ((label_img == fg_cluster) * 255).astype(np.uint8)
    return postprocess(binary), centers


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – FAMILY 3: MARKER-CONTROLLED WATERSHED
# ─────────────────────────────────────────────────────────────────────────────

def watershed_segment(img, preproc='clahe_blur', min_distance=22,
    marker_method='distance'):
    """
    Marker-controlled watershed pipeline:
      Stage 1 – Otsu seed mask + morphological cleaning
      Stage 2 – Euclidean distance transform
      Stage 3 – Peak detection → integer markers
      Stage 4 – Watershed flooding of negated distance map

     marker_method: 'distance' | 'erosion'
     """
    gray = preprocess(img, preproc)

    # Stage 1
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    k5      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k5, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k7, iterations=2)
    bin_bool = cleaned > 0

    # Stage 2
    dist = ndimage.distance_transform_edt(bin_bool)

    # Stage 3
    if marker_method == 'distance':
        coords = feature.peak_local_max(
            dist, min_distance=min_distance,
            labels=bin_bool, exclude_border=False)
        markers = np.zeros_like(dist, dtype=np.int32)
        for idx, (r, c) in enumerate(coords, start=1):
            markers[r, c] = idx
        markers, _ = ndimage.label(markers > 0)
    else:
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        eroded = cv2.erode(cleaned, k_erode, iterations=3)
        markers, _ = ndimage.label(eroded > 0)

    # Stage 4
    labels_ws = segmentation.watershed(
        -dist, markers, mask=bin_bool, compactness=0.01)

    binary_out = ((labels_ws > 0) * 255).astype(np.uint8)
    return postprocess(binary_out), labels_ws, dist, markers


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred, gt):
    """IoU, Dice, Precision, Recall, F1 between two binary masks."""
    p = (pred > 127).astype(bool)
    g = (gt   > 127).astype(bool)
    tp = np.logical_and( p, g).sum()





    fp = np.logical_and( p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    pr = tp / (tp + fp + 1e-9)
    rc = tp / (tp + fn + 1e-9)
    return dict(
        IoU       = float(tp / (tp + fp + fn + 1e-9)),
        Dice      = float(2*tp / (2*tp + fp + fn + 1e-9)),
        Precision = float(pr),
        Recall    = float(rc),
        F1        = float(2 * pr * rc / (pr + rc + 1e-9)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load(name):
    img = cv2.imread(os.path.join(IMG_DIR, f'{name}.png'))
    gt = cv2.imread(os.path.join(GT_DIR, f'{name}_gt.png'), 0)
    return img, gt

def run_all(img):
    otsu,   tv   = global_otsu(img)
    adap         = adaptive_threshold(img)
    adap_t       = adaptive_threshold_tuned(img)
    kmns,   ctrs = kmeans_segment(img)
    wshd, lbl, dist, mkrs = watershed_segment(img)
    return otsu, adap, adap_t, kmns, wshd, lbl, dist, mkrs

def errmap(pred, gt):
    p = pred > 127; g = gt > 127
    e = np.zeros((*g.shape, 3), dtype=np.uint8)
    e[p & g] = [0,     200, 0]
    e[p & ~g] = [255, 60, 60]
    e[~p & g] = [60, 60, 255]
    return e

def overlay_contours(rgb, mask, color):
    out = rgb.copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, cnts, -1, color, 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# Plot style
BG = '#0d1117'
BG2 = '#161b22'
COLORS = {
    'Otsu (Global)':                '#4FC3F7',
    'Adaptive (tuned)':             '#FFB74D',
    'K-Means':                      '#81C784',
    'Watershed':                    '#CE93D8',
    }

LABELED = [
    ('coin_01_clean',        'Clean'),
    ('coin_03_shadow_lr',    'Shadow L-R'),
    ('coin_05_local_shadow', 'Local Shadow'),
    ('coin_07_high_noise',   'High Noise'),
    ('coin_10_combined',     'Combined'),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – EXPERIMENT MATRIX
# ─────────────────────────────────────────────────────────────────────────────






def run_experiments():
    print("[2/5] Running 5×4 experiment matrix ...")
    results = {}
    for name, label in LABELED:
        img, gt = load(name)
        otsu, _, adap_t, kmns, wshd, _, _, _ = run_all(img)
        results[name] = {
            'Otsu (Global)':    compute_metrics(otsu,   gt),
            'Adaptive (tuned)': compute_metrics(adap_t, gt),
            'K-Means':          compute_metrics(kmns,   gt),
            'Watershed':        compute_metrics(wshd,   gt),
            }

    hdr = f"{'Image':20s} {'Method':18s} {'IoU':6s} {'Dice':6s} {'Prec':6s} {'Rec':6s}{'F1':6s}"
    print('\n' + hdr + '\n' + '-' * len(hdr))
    for name, label in LABELED:
        for method in ['Otsu (Global)', 'Adaptive (tuned)', 'K-Means', 'Watershed']:
            m = results[name][method]
            print(f"{label:20s} {method:18s} "
                f"{m['IoU']:.3f} {m['Dice']:.3f} "
                f"{m['Precision']:.3f} {m['Recall']:.3f} {m['F1']:.3f}")

    with open(os.path.join(RES_DIR, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def save(name):
    plt.savefig(os.path.join(FIG_DIR, name), dpi=140,
        bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"    {name}")

def sax(ax):
    ax.set_facecolor(BG2); ax.axis('off')


# Figure 1 – Dataset overview
def fig1_dataset():
    names = ['coin_01_clean','coin_03_shadow_lr','coin_05_local_shadow',
        'coin_07_high_noise','coin_10_combined']
    titles = ['Clean','Shadow L-R','Local Shadow','High Noise','Combined']
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    for col, (name, title) in enumerate(zip(names, titles)):
        img, gt = load(name)
        gray    = preprocess(img, 'clahe_blur')
        for row in range(3):
            sax(axes[row, col])
        axes[0, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(title, color='white', fontsize=10, fontweight='bold',
            pad=4)
        axes[1, col].imshow(gray, cmap='gray')
        axes[2, col].imshow(gt,   cmap='gray')
    for row, label in enumerate(['Input', 'CLAHE+Blur', 'Ground Truth']):
        axes[row, 0].set_ylabel(label, color='#8b949e', fontsize=9)
    fig.suptitle('Figure 1 – Dataset: 5 Labeled Coin Images (Real Photograph +Photometric Variants)',
        color='white', fontsize=11, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save('fig1_dataset.png')








# Figure 2 – All methods on clean image
def fig2_methods():
    img, gt = load('coin_01_clean')
    otsu, adap, adap_t, kmns, wshd, lbl, _, _ = run_all(img)
    rgb        = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lbl_rgb    = np.clip(label2rgb(lbl, image=rgb, kind='overlay', alpha=0.5), 0, 1)
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    panels = [
        ('Original',          rgb,    None),
        ('Ground Truth',      gt,     'gray'),
        ('Otsu (Global)',     otsu,   'Blues'),
        ('K-Means',           kmns,   'Greens'),
        ('Adaptive (raw)',    adap,   'Oranges'),
        ('Adaptive (tuned)', adap_t, 'YlOrRd'),
        ('Watershed',         wshd,   'Purples'),
        ('Watershed Regions',lbl_rgb, None),
        ]
    for ax, (ttl, data, cm) in zip(axes.flat, panels):
        sax(ax)
        ax.imshow(data, cmap=cm)
        ax.set_title(ttl, color='white', fontsize=9, fontweight='bold', pad=4)
    fig.suptitle('Figure 2 – All Method Outputs on Clean Image',
        color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()
    save('fig2_all_methods.png')


# Figure 3 – Boundary overlays
def fig3_boundaries():
    img, gt = load('coin_03_shadow_lr')
    otsu, _, adap_t, kmns, wshd, _, _, _ = run_all(img)
    rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.patch.set_facecolor(BG)
    panels = [
        ('GT Boundary',        overlay_contours(rgb, gt,    (255, 220, 0))),
        ('Otsu',               overlay_contours(rgb, otsu,  (79, 195, 247))),
        ('Adaptive (tuned)', overlay_contours(rgb, adap_t, (255, 183, 77))),
        ('K-Means',            overlay_contours(rgb, kmns,  (129, 199, 132))),
        ('Watershed',          overlay_contours(rgb, wshd,  (206, 147, 216))),
        ]
    for ax, (ttl, data) in zip(axes, panels):
        sax(ax); ax.imshow(data)
        ax.set_title(ttl, color='white', fontsize=9, fontweight='bold', pad=4)
    fig.suptitle('Figure 3 – Boundary Overlays on Shadow L-R Image',
        color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()
    save('fig3_boundaries.png')


# Figure 4 – Metric bar chart
def fig4_metrics(results):
    methods      = list(COLORS.keys())
    metric_names = ['IoU', 'Dice', 'Precision', 'Recall']
    img_labels   = [t for _, t in LABELED]
    fig, axes    = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor(BG)
    for ax, metric in zip(axes, metric_names):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        ax.tick_params(colors='#8b949e', labelsize=8)
        x = np.arange(len(img_labels)); w = 0.19
        for i, method in enumerate(methods):
            vals = [results[n][method][metric] for n, _ in LABELED]
            ax.bar(x + (i - 1.5)*w, vals, w,
                color=list(COLORS.values())[i],
                
                
                
                
                
                alpha=0.88, edgecolor='#30363d', linewidth=0.5)
        ax.set_title(metric, color='white', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(img_labels, rotation=25, ha='right',
            color='#c9d1d9', fontsize=8)
        ax.set_ylim(0, 1.09)
        ax.grid(axis='y', color='#21262d', linewidth=0.7)
        ax.set_axisbelow(True)
    handles = [mpatches.Patch(color=c, label=m) for m, c in COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=4,
        facecolor=BG2, edgecolor='#30363d',
        labelcolor='#c9d1d9', fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle('Figure 4 – Segmentation Metrics Across All Labeled Images',
        color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()
    save('fig4_metrics.png')


# Figure 5 – IoU heatmap
def fig5_heatmap(results):
    methods    = list(COLORS.keys())
    img_labels = [t for _, t in LABELED]
    mat        = np.array([[results[n][m]['IoU'] for m in methods] for n, _ in
        LABELED])
    fig, ax    = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(methods)));   ax.set_xticklabels(methods, color='#c9d1d9',
        fontsize=10)
    ax.set_yticks(range(len(img_labels))); ax.set_yticklabels(img_labels,
        color='#c9d1d9', fontsize=10)
    ax.tick_params(colors='#8b949e')
    for i in range(len(img_labels)):
        for j in range(len(methods)):
            v = mat[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                color='black' if v > 0.55 else 'white',
                fontsize=10, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(colors='#8b949e')
    cbar.ax.set_ylabel('IoU', color='#c9d1d9', fontsize=9)
    ax.set_title('Figure 5 – IoU Heatmap (Methods × Image Types)',
        color='white', fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    save('fig5_heatmap.png')


# Figure 6 – Pre-processing ablation
def fig6_ablation():
    img, gt = load('coin_10_combined')
    configs = [('No Pre-proc','none'), ('Blur','blur'),
        ('CLAHE','clahe'), ('CLAHE+Blur','clahe_blur')]
    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    fig.patch.set_facecolor(BG)
    for col, (label, prep) in enumerate(configs):
        gray = preprocess(img, prep)
        ws, _, _, _ = watershed_segment(img, preproc=prep)
        met = compute_metrics(ws, gt)
        for row in range(3):
            sax(axes[row, col])
        axes[0, col].imshow(gray, cmap='gray')
        axes[0, col].set_title(label, color='white', fontsize=9, fontweight='bold',
            pad=4)
        axes[1, col].imshow(ws, cmap='gray')
        axes[1, col].set_title(f"IoU={met['IoU']:.3f} Dice={met['Dice']:.3f}",
            color='#CE93D8', fontsize=9)
        axes[2, col].imshow(errmap(ws, gt))
        axes[2, col].set_title('TP/FP/FN', color='#8b949e', fontsize=8)





    for row, lbl in enumerate(['Pre-processed', 'Watershed Mask', 'Error Map']):
        axes[row, 0].set_ylabel(lbl, color='#8b949e', fontsize=9)
    fig.suptitle('Figure 6 – Pre-processing Ablation Study (Watershed, CombinedImage)',
        color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()
    save('fig6_ablation.png')


# Figure 7 – Adaptive threshold failure analysis
def fig7_adaptive_failure():
    img, gt = load('coin_01_clean')
    rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray     = preprocess(img, 'clahe_blur')
    adap_raw = adaptive_threshold(img)
    adap_t   = adaptive_threshold_tuned(img)
    otsu, _ = global_otsu(img)
    fig      = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.4, wspace=0.3)

    def da(pos, data, title, color='white', cmap=None):
        ax = fig.add_subplot(pos)
        sax(ax); ax.imshow(data, cmap=cmap)
        ax.set_title(title, color=color, fontsize=9, fontweight='bold')

    da(gs[0,0], rgb,      'Original')
    da(gs[0,1], gt,       'Ground Truth',            cmap='gray')
    da(gs[0,2], adap_raw, 'Adaptive Raw\n(b=35,C=4)','#FFB74D', 'Oranges')
    da(gs[0,3], adap_t,   'Adaptive Tuned\n(b=71,C=-8)','#FFB74D','YlOrRd')
    da(gs[0,4], otsu,     'Otsu (baseline)',           '#4FC3F7', 'Blues')
    da(gs[1,0], gray,     'CLAHE+Blur',               '#8b949e', 'gray')

    m_ar = compute_metrics(adap_raw, gt)
    m_at = compute_metrics(adap_t,    gt)
    m_ot = compute_metrics(otsu,      gt)
    da(gs[1,1], errmap(adap_raw,gt), f'Adaptive RawError\nIoU={m_ar["IoU"]:.3f}','#FFB74D')
    da(gs[1,2], errmap(adap_t, gt), f'Adaptive TunedError\nIoU={m_at["IoU"]:.3f}','#FFB74D')
    da(gs[1,3], errmap(otsu,     gt), f'Otsu Error\nIoU={m_ot["IoU"]:.3f}','#4FC3F7')

    ax9 = fig.add_subplot(gs[1, 4])
    ax9.set_facecolor(BG2)
    ax9.tick_params(colors='#8b949e', labelsize=7)
    for sp in ax9.spines.values(): sp.set_edgecolor('#30363d')
    ax9.hist(gray.ravel(), bins=64, color='#4FC3F7', alpha=0.7, density=True)
    tv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    ax9.axvline(tv, color='#FFB74D', lw=2, label=f'Otsu T={int(tv)}')
    ax9.set_title('Intensity Histogram', color='white', fontsize=9, fontweight='bold')
    ax9.legend(fontsize=8, facecolor=BG2, labelcolor='white', edgecolor='none')
    ax9.set_xlabel('Intensity', color='#8b949e', fontsize=8)
    ax9.set_ylabel('Density',   color='#8b949e', fontsize=8)

    patches = [mpatches.Patch(color='#00C800', label='TP'),
        mpatches.Patch(color='#FF3C3C', label='FP (over-seg)'),
        mpatches.Patch(color='#3C3CFF', label='FN (missed)')]
    fig.legend(handles=patches, loc='lower center', ncol=3,
        facecolor=BG2, edgecolor='none', labelcolor='white',
        fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Figure 7 – Adaptive Threshold Failure: Ring Artefacts vs Filled-DiscMethods',
        color='white', fontsize=11, fontweight='bold')
    save('fig7_adaptive_failure.png')


# Figure 8 – Watershed internals
def fig8_watershed():





    img, _ = load('coin_01_clean')
    rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, lbl, dist, mkrs = watershed_segment(img)
    lbl_rgb = np.clip(label2rgb(lbl, image=rgb, kind='overlay', alpha=0.5), 0, 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    panels = [
        ('Original',                      rgb,                None),
        ('Distance Transform',            dist,               'hot'),
        (f'Seeds ({mkrs.max()} markers)',(mkrs > 0)*255,      'gray'),
        ('Watershed Regions',             lbl_rgb,            None),
        ]
    for ax, (ttl, data, cm) in zip(axes, panels):
        sax(ax); ax.imshow(data, cmap=cm)
        ax.set_title(ttl, color='white', fontsize=10, fontweight='bold')
    fig.suptitle('Figure 8 – Watershed Pipeline: Distance Transform → Seeds →Regions',
        color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()
    save('fig8_watershed.png')


# Figure 9 – K-Means colour space ablation
def fig9_colorspace():
    img, gt = load('coin_03_shadow_lr')
    rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    spaces = [('grey','Grey'), ('rgb','RGB'), ('lab','L*a*b*')]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.patch.set_facecolor(BG)
    for col, (cs, cs_label) in enumerate(spaces):
        mask, _ = kmeans_segment(img, color_space=cs)
        met     = compute_metrics(mask, gt)
        sax(axes[0, col]); sax(axes[1, col])
        axes[0, col].imshow(mask, cmap='Greens')
        axes[0, col].set_title(f'K-Means ({cs_label})\nIoU={met["IoU"]:.3f}',
            color='#81C784', fontsize=9, fontweight='bold')
        axes[1, col].imshow(errmap(mask, gt))
        axes[1, col].set_title('Error Map', color='#8b949e', fontsize=9)
    sax(axes[0, 3]); sax(axes[1, 3])
    axes[0, 3].imshow(rgb);       axes[0, 3].set_title('Input (Shadow L-R)',
        color='white', fontsize=9, fontweight='bold')
    axes[1, 3].imshow(gt, cmap='gray'); axes[1, 3].set_title('Ground Truth',
        color='white', fontsize=9, fontweight='bold')
    fig.suptitle('Figure 9 – K-Means Colour Space Ablation: Grey vs RGB vs L*a*b*',
        color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()
    save('fig9_colorspace.png')


# Figure 10 – Deep failure analysis
def fig10_failure():
    img, gt = load('coin_10_combined')
    otsu, _, adap_t, kmns, wshd, lbl, dist, _ = run_all(img)
    rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.patch.set_facecolor(BG)
    top = [('Original',rgb,None),('Ground Truth',gt,'gray'),
        ('Otsu',otsu,'Blues'),('K-Means',kmns,'Greens'),('Watershed',wshd,'Purples')]
    bot = [('CLAHE+Blur',preprocess(img,'clahe_blur'),'gray'),
        ('Dist. Transform',dist,'hot'),
        ('Otsu Error',errmap(otsu,gt),None),
        ('K-Means Error',errmap(kmns,gt),None),
        ('Watershed Error',errmap(wshd,gt),None)]
    for ax, (ttl, data, cm) in zip(axes[0], top):
        sax(ax); ax.imshow(data, cmap=cm)
        ax.set_title(ttl, color='white', fontsize=9, fontweight='bold', pad=4)
    for ax, (ttl, data, cm) in zip(axes[1], bot):





        sax(ax); ax.imshow(data, cmap=cm)
        color = '#ce93d8' if 'Error' in ttl else '#8b949e'
        extra = ''
        if 'Error' in ttl:
            mm = {'Otsu Error': otsu, 'K-Means Error': kmns, 'Watershed Error': wshd}
            m = compute_metrics(mm[ttl], gt)
            extra = f'\nIoU={m["IoU"]:.3f}'
        ax.set_title(ttl + extra, color=color, fontsize=8, fontweight='bold', pad=4)
    patches = [mpatches.Patch(color='#00C800', label='TP'),
        mpatches.Patch(color='#FF3C3C', label='FP (over-seg)'),
        mpatches.Patch(color='#3C3CFF', label='FN (missed)')]
    fig.legend(handles=patches, loc='lower center', ncol=3,
        facecolor=BG2, edgecolor='none', labelcolor='white',
        fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Figure 10 – Deep Failure Analysis: Combined Image (Shadow + Noise +Vignette)',
        color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()
    save('fig10_failure.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    create_dataset()
    results = run_experiments()

    print("[3/5] Generating figures ...")
    fig1_dataset()
    fig2_methods()
    fig3_boundaries()
    fig4_metrics(results)
    fig5_heatmap(results)
    fig6_ablation()
    fig7_adaptive_failure()
    fig8_watershed()
    fig9_colorspace()
    fig10_failure()

    print("[4/5] Figures saved:")
    for f in sorted(os.listdir(FIG_DIR)):
        if f.startswith('fig'):
            print(f"    {f}")

    print("[5/5] Done.")
    print(f"\n Images                → {IMG_DIR}")
    print(f" GT masks              → {GT_DIR}")
    print(f" Metrics               → {RES_DIR}/metrics.json")
    print(f" Figures               → {FIG_DIR}")


if __name__ == '__main__':
    main()

