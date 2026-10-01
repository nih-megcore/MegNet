"""Compatibility helpers for custom topomap rendering.

These helpers use Matplotlib's public artist API in place of private MNE
helpers that were removed in MNE 1.13.
"""

from matplotlib.patches import Ellipse, Polygon


def make_clip_patch(outlines, extrapolate, interp, axes):
    """Create the clipping patch used by MEGnet's custom topomap."""
    clip_radius = outlines["clip_radius"]
    clip_origin = outlines.get("clip_origin", (0.0, 0.0))
    use_default_outlines = any(key.startswith("head") for key in outlines)
    patch = None

    if "patch" in outlines:
        patch = outlines["patch"]
        patch = patch() if callable(patch) else patch
        patch.set_clip_on(False)
        axes.add_patch(patch)
        axes.set_transform(axes.transAxes)
        axes.set_clip_path(patch)

    if use_default_outlines:
        if extrapolate == "local":
            patch = Polygon(
                interp.mask_pts,
                clip_on=True,
                transform=axes.transData,
            )
        else:
            patch = Ellipse(
                clip_origin,
                2 * clip_radius[0],
                2 * clip_radius[1],
                clip_on=True,
                transform=axes.transData,
            )

    return patch


def set_contour_clip_path(contour, patch):
    """Apply a clip path across supported Matplotlib contour APIs."""
    if hasattr(contour, "set_clip_path"):
        contour.set_clip_path(patch)
        return

    for collection in contour.collections:
        collection.set_clip_path(patch)
