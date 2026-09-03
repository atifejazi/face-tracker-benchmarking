"""Shared bbox-based face cropping for MultiREX benchmark videos."""

import numpy as np
from skimage.transform import estimate_transform, warp


def bbox_to_similarity_transform(bbox, scale=1.25, image_size=224, deca_style=False):
    """Build a similarity transform from a MultiREX bbox [x_min, y_min, x_max, y_max]."""
    left, top, right, bottom = bbox
    old_size = (right - left + bottom - top) / 2.0
    center = np.array([
        right - (right - left) / 2.0,
        bottom - (bottom - top) / 2.0,
    ])
    if deca_style:
        center[1] += old_size * 0.12

    size = int(old_size * scale)
    src_pts = np.array([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2],
    ])
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    return estimate_transform("similarity", src_pts, dst_pts)


def crop_frame_with_bbox(frame, bbox, scale=1.25, image_size=224, deca_style=False):
    """Crop and warp a BGR frame to a square face image."""
    tform = bbox_to_similarity_transform(bbox, scale=scale, image_size=image_size, deca_style=deca_style)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    cropped = warp(
        frame,
        tform.inverse,
        output_shape=(image_size, image_size),
        preserve_range=True,
    ).astype(np.uint8)
    return cropped, tform
