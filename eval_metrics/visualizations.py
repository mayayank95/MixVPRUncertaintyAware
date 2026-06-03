import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
from PIL import Image, ImageOps
import torchvision.transforms as tfm
from pathlib import Path

logger = logging.getLogger(__name__)

# Height and width of a single image for visualization
IMG_HW = 512
TEXT_H = 80
FONTSIZE = 30
OVERLAY_FONTSIZE = 20
SPACE = 50  # Space between two images

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _get_font(size):
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def write_labels_to_image(labels=["text1", "text2"]):
    """Creates an image with text labels (Query, Pred0 - True/False, ...)."""
    font = _get_font(FONTSIZE)
    img = Image.new("RGB", ((IMG_HW * len(labels)) + SPACE * (len(labels) - 1), TEXT_H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(labels):
        _, _, w, h = d.textbbox((0, 0), text, font=font)
        x = (IMG_HW + SPACE) * i + IMG_HW // 2 - w // 2
        d.text((x, (TEXT_H - h) // 2), text, fill=(0, 0, 0), font=font)
    return img


def _draw_overlay(img, lines):
    """Draw semi-transparent overlay with text lines at the top of a PIL image."""
    if not lines:
        return img
    font = _get_font(OVERLAY_FONTSIZE)
    draw = ImageDraw.Draw(img)
    line_h = OVERLAY_FONTSIZE + 4
    strip_h = line_h * len(lines) + 6
    # Semi-transparent dark strip
    overlay = Image.new("RGBA", (img.width, strip_h), (0, 0, 0, 160))
    img.paste(Image.alpha_composite(Image.new("RGBA", overlay.size, (0, 0, 0, 0)), overlay).convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((6, 3 + i * line_h), text, fill=(255, 255, 255), font=font)
    return img


def draw_box(img, c=(0, 1, 0), thickness=20):
    """Draw a colored box around an image. Image should be a PIL.Image."""
    assert isinstance(img, Image.Image)
    img = tfm.ToTensor()(img)
    assert len(img.shape) >= 2, f"{img.shape=}"
    c = torch.tensor(c).type(torch.float).reshape(3, 1, 1)
    img[..., :thickness, :] = c
    img[..., -thickness:, :] = c
    img[..., :, -thickness:] = c
    img[..., :, :thickness] = c
    return tfm.ToPILImage()(img)


def build_prediction_image(images_paths, preds_correct, distances=None,
                           query_variance=None, pred_variances=None,
                           gt_images_paths=None, gt_distances=None, gt_variances=None):
    """Build a row of images, where the first is the query and the rest are predictions.
    For each image, if is_correct then draw a green/red box.
    Overlays variance and distance text directly on each image.
    distances: optional list of L2 distances for each prediction (after Query).
    query_variance: optional float, query's mean variance (uncertainty).
    pred_variances: optional list of floats, per-prediction db item mean variance.
    """
    assert len(images_paths) == len(preds_correct)
    # Clean labels (no variance/distance — those go on the overlay)
    labels = ["Query"]
    for i, is_correct in enumerate(preds_correct[1:]):
        if is_correct is None:
            labels.append(f"Pred{i}")
        else:
            labels.append(f"Pred{i} - {'✓' if is_correct else '✗'}")

    images = [Image.open(path).convert("RGB") for path in images_paths]
    for img_idx, (img, is_correct) in enumerate(zip(images, preds_correct)):
        if is_correct is None:
            continue
        color = (0, 1, 0) if is_correct else (1, 0, 0)
        img = draw_box(img, color)
        images[img_idx] = img

    resized_images = [ImageOps.pad(img.resize((IMG_HW, IMG_HW)), (IMG_HW, IMG_HW), color='white') for img in images]

    # Draw overlays on each image
    for idx, img in enumerate(resized_images):
        overlay_lines = []
        if idx == 0:
            # Query image: show query variance
            if query_variance is not None:
                overlay_lines.append(f"unc: {query_variance:.4f}")
        else:
            # Prediction image: show db variance and distance
            pred_idx = idx - 1
            if pred_variances is not None and pred_idx < len(pred_variances):
                overlay_lines.append(f"unc: {pred_variances[pred_idx]:.4f}")
            if distances is not None and pred_idx < len(distances):
                overlay_lines.append(f"dist: {distances[pred_idx]:.4f}")
        resized_images[idx] = _draw_overlay(img, overlay_lines)

    total_w = len(resized_images) * IMG_HW + max(0, len(resized_images) - 1) * SPACE
    concat_image = Image.new('RGB', (total_w, IMG_HW), (255, 255, 255))
    x = 0
    for img in resized_images:
        concat_image.paste(img, (x, 0))
        x += IMG_HW + SPACE

    try:
        labels_image = write_labels_to_image(labels)
        final_image = Image.fromarray(np.concatenate((np.array(labels_image), np.array(concat_image)), axis=0))
    except OSError:
        final_image = concat_image

    if gt_images_paths:
        gt_labels = [""] + [f"GT{i}" for i in range(len(gt_images_paths))]
        gt_images = [Image.new('RGB', (IMG_HW, IMG_HW), (255, 255, 255))]
        for path in gt_images_paths:
            img = Image.open(path).convert("RGB")
            img = draw_box(img, (0, 0, 1))
            gt_images.append(img)
            
        gt_resized = [ImageOps.pad(img.resize((IMG_HW, IMG_HW)), (IMG_HW, IMG_HW), color='white') for img in gt_images]
        
        for idx in range(1, len(gt_resized)):
            overlay_lines = []
            gt_idx = idx - 1
            if gt_variances is not None and gt_idx < len(gt_variances):
                overlay_lines.append(f"unc: {gt_variances[gt_idx]:.4f}")
            if gt_distances is not None and gt_idx < len(gt_distances):
                overlay_lines.append(f"dist: {gt_distances[gt_idx]:.4f}")
            gt_resized[idx] = _draw_overlay(gt_resized[idx], overlay_lines)
            
        row2_w = len(gt_resized) * IMG_HW + max(0, len(gt_resized) - 1) * SPACE
        row2_image = Image.new('RGB', (max(total_w, row2_w), IMG_HW), (255, 255, 255))
        x = 0
        for img in gt_resized:
            row2_image.paste(img, (x, 0))
            x += IMG_HW + SPACE
            
        try:
            # Create a label strip for GT wide enough
            gt_labels_img_base = write_labels_to_image(gt_labels)
            gt_labels_image = Image.new('RGB', (max(total_w, row2_w), TEXT_H), (255, 255, 255))
            gt_labels_image.paste(gt_labels_img_base, (0, 0))
            
            row2_combined = np.concatenate((np.array(gt_labels_image), np.array(row2_image)), axis=0)
            
            # Ensure final_image and row2_combined have the same width
            if final_image.width < row2_combined.shape[1]:
                new_fi = Image.new('RGB', (row2_combined.shape[1], final_image.height), (255, 255, 255))
                new_fi.paste(final_image, (0, 0))
                final_image = new_fi
                
            final_image = Image.fromarray(np.concatenate((np.array(final_image), row2_combined), axis=0))
        except OSError:
            pass

    return final_image


def save_file_with_paths(query_path, preds_paths, positives_paths, output_path, use_labels=True, distances=None,
                         query_variance=None):
    file_content = []
    file_content.append("Query path:")
    file_content.append(query_path)
    if query_variance is not None:
        file_content.append(f"Query uncertainty: {query_variance:.4f}")
    file_content.append("\nPredictions paths:")
    for i, p in enumerate(preds_paths):
        dist_str = f"  (dist: {distances[i]:.4f})" if distances is not None else ""
        file_content.append(f"{p}{dist_str}")
    file_content.append("\n")
    if use_labels:
        file_content.append("Positives paths:")
        file_content.append("\n".join(positives_paths) + "\n")
    with open(output_path, "w") as file:
        _ = file.write("\n".join(file_content))


def save_preds(predictions, eval_ds, log_dir, save_only_wrong_preds=None, use_labels=True, num_preds_to_viz=None,
               distances=None, query_variances=None, db_variances=None, q_desc=None, db_desc=None):
    """For each query, save an image containing the query and its predictions,
    and a file with the paths of the query, its predictions and its positives.

    Parameters
    ----------
    predictions : np.array of shape [num_queries x num_preds_to_viz], with the preds
        for each query
    eval_ds : TestDataset
    log_dir : Path with the path to save the predictions
    save_only_wrong_preds : bool, if True save only the wrongly predicted queries,
        i.e. the ones where the first pred is uncorrect (further than 25 m)
    distances : np.array of shape [num_queries x num_preds], optional. L2 distance per prediction.
    query_variances : np.array of shape [num_queries], optional. Mean variance (uncertainty) per query.
    db_variances : np.array of shape [num_database, D], optional. Variance vectors for each database item.
    """
    if use_labels:
        positives_per_query = eval_ds.get_positives()

    viz_dir = Path(f"{log_dir}/preds")
    viz_dir.mkdir(exist_ok=True)
    logger.debug(f"Saving predictions in {viz_dir}")
    for query_index, preds in enumerate(tqdm(predictions[:num_preds_to_viz], desc="Saving preds")):
        query_path = eval_ds.queries_paths[query_index]
        list_of_images_paths = [query_path]
        # List of None (query), True (correct preds) or False (wrong preds)
        preds_correct = [None]
        for pred_index, pred in enumerate(preds):
            pred_path = eval_ds.database_paths[pred]
            list_of_images_paths.append(pred_path)
            if use_labels:
                is_correct = pred in positives_per_query[query_index]
            else:
                is_correct = None
            preds_correct.append(is_correct)

        if save_only_wrong_preds and preds_correct[1]:
            continue

        query_dists = distances[query_index].tolist() if distances is not None else None
        gt_images_paths = None
        gt_distances = None
        gt_variances = None

        if use_labels:
            gt_indices = positives_per_query[query_index][:3] # up to 3 GT images
            if len(gt_indices) > 0:
                gt_images_paths = [eval_ds.database_paths[idx] for idx in gt_indices]
                if q_desc is not None and db_desc is not None:
                    q_feat = q_desc[query_index]
                    
                    # Ensure gt_indices are within range of db_desc (e.g. for dry runs with sliced DB)
                    valid_gt_mask = (gt_indices < len(db_desc))
                    if np.any(valid_gt_mask):
                        valid_gt_indices = gt_indices[valid_gt_mask]
                        gt_feats = db_desc[valid_gt_indices]
                        # FAISS IndexFlatL2 returns squared Euclidean distance. 
                        # Compute squared Euclidean distance for GT images to match.
                        gt_distances = [float(np.linalg.norm(q_feat - f)**2) for f in gt_feats]
                    else:
                        gt_distances = None
                if db_variances is not None:
                    # Filter for db_variances as well
                    valid_gt_mask = (gt_indices < len(db_variances))
                    if np.any(valid_gt_mask):
                        gt_variances = [float(np.mean(db_variances[idx])) for idx in gt_indices[valid_gt_mask]]
                    else:
                        gt_variances = None

        q_var = float(query_variances[query_index]) if query_variances is not None else None
        # Per-prediction database item mean variances
        pred_vars = None
        if db_variances is not None:
            pred_vars = [float(np.mean(db_variances[p])) for p in preds]
        prediction_image = build_prediction_image(list_of_images_paths, preds_correct, distances=query_dists,
                                                 query_variance=q_var, pred_variances=pred_vars,
                                                 gt_images_paths=gt_images_paths, gt_distances=gt_distances,
                                                 gt_variances=gt_variances)
        pred_image_path = viz_dir / f"{query_index:03d}.jpg"
        prediction_image.save(pred_image_path)

        if use_labels:
            positives_paths = [eval_ds.database_paths[idx] for idx in positives_per_query[query_index]]
        else:
            positives_paths = None
        save_file_with_paths(
            query_path=list_of_images_paths[0],
            preds_paths=list_of_images_paths[1:],
            positives_paths=positives_paths,
            output_path=viz_dir / f"{query_index:03d}.txt",
            use_labels=use_labels,
            distances=query_dists,
            query_variance=q_var,
        )

def main():
    """
    End-to-end test for save_preds using a mock dataset.
    """
    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------
    root = Path("save_preds_debug")
    root.mkdir(exist_ok=True)

    img_dir = root / "images"
    img_dir.mkdir(exist_ok=True)

    # --------------------------------------------------
    # Create dummy images
    # --------------------------------------------------
    def make_img(color, path):
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
        arr[..., color] = 255
        Image.fromarray(arr).save(path)

    # Queries
    queries_paths = []
    for i in range(3):
        p = img_dir / f"query_{i}.jpg"
        make_img(i % 3, p)
        queries_paths.append(str(p))

    # Database images
    database_paths = []
    for i in range(6):
        p = img_dir / f"db_{i}.jpg"
        make_img((i + 1) % 3, p)
        database_paths.append(str(p))

    print("✔ Dummy images created")

    # --------------------------------------------------
    # Mock dataset
    # --------------------------------------------------
    class MockEvalDataset:
        def __init__(self, queries_paths, database_paths):
            self.queries_paths = queries_paths
            self.database_paths = database_paths

        def get_positives(self):
            # Each query has two positives
            return {
                0: [0, 1],
                1: [2, 3],
                2: [4, 5],
            }

    eval_ds = MockEvalDataset(queries_paths, database_paths)

    # --------------------------------------------------
    # Fake predictions: shape [num_queries, num_preds]
    # --------------------------------------------------
    predictions = np.array([
        [0, 2, 3],
        [3, 4, 1],
        [5, 0, 2],
    ])

    # --------------------------------------------------
    # Run save_preds
    # --------------------------------------------------
    log_dir = root
    save_preds(
        predictions=predictions,
        eval_ds=eval_ds,
        log_dir=log_dir,
        save_only_wrong_preds=False,
        use_labels=True,
        num_preds_to_viz=3,
    )

    print("\n✔ save_preds finished successfully")
    print(f"Check results in: {root.resolve()}")


if __name__ == "__main__":
    main()
  