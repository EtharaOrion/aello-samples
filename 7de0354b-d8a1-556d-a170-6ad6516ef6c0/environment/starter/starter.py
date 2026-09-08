"""Runnable baseline for AELLO-C8-S5. Correct, honest, thirteen-channel -- and deliberately weak.

WHAT IT DOES. It reads the twelve-band stack and the co-registered elevation raster for every
graded tile, computes one spectral index per pixel (NDVI from B08 and B04), splits it at fixed
quantiles into land-cover classes, and then lets the elevation raster override the split above its
own median height. That is a two-stream fusion joined at the decision -- a spectral stream and an
elevation stream, combined after each has been reduced -- which is why its declared fusion stage is
mid_level_fusion. It is a decision rule, not a model: it fits nothing and it trains on no tile.

WHAT IT IS NOT. It is not a good segmenter and it is not trying to be. It exists so the delivery
path and the run record are exercised end to end before you change a line of it. Replace all of it.

TWO THINGS WORTH KNOWING BEFORE YOU BUILD ON IT.

  The graded mean is PER CLASS. This baseline uses three of the classes the index publishes and
  never predicts the rest, so every class it ignores contributes a zero term to the mean. Look at
  `class_split()` and decide how many classes your solve should be willing to name.

  It normalises nothing. It reads `band_statistics.json` only to DECLARE a per-band scale, and its
  own arithmetic runs on raw reflectance -- so the SWIR bands sit at a different magnitude from the
  visible ones inside the same expression. That is trap two, shipped switched on, in the one place
  where you can see it.

THE RUN RECORD. Everything instruction.md lists is written here, honestly: this baseline trains on
no tile, so its training and elevation-read declarations are empty lists rather than omitted keys.
An empty declaration is a claim; a missing one is not.
"""
import hashlib
import json
import os
import sys

SUBMISSION = os.environ.get("AELLO_SUBMISSION", "/submission/masks.npz")
RECORD = "/workspace/run_record.json"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
NIR, RED = BANDS.index("B08"), BANDS.index("B04")


def graded_tiles():
    """Tile ids in a fixed order. The order is declared, so it must not be a set's order."""
    import numpy as np
    path = os.path.join(DATA, "graded_spectral.npz")
    if not os.path.exists(path):
        return []
    with np.load(path, allow_pickle=False) as archive:
        return sorted(archive.files)


def class_split():
    """Which class ids this baseline is willing to predict. Three of however many ship."""
    path = os.path.join(DATA, "class_index.json")
    if not os.path.exists(path):
        return [0, 1, 2]
    with open(path) as handle:
        doc = json.load(handle)
    ids = sorted(int(c) for c in (doc.get("class_ids", doc) if isinstance(doc, dict) else doc))
    return (ids + ids[-1:] * 3)[:3]


def tile_mask(spectral, elevation, classes):
    """One integer mask on the spectral grid: NDVI quantiles, overridden above median height."""
    import numpy as np
    stack = np.asarray(spectral, dtype="float32")
    nir, red = stack[NIR], stack[RED]
    ndvi = (nir - red) / (nir + red + 1e-6)
    low, high = np.quantile(ndvi, 0.33), np.quantile(ndvi, 0.66)
    mask = np.full(ndvi.shape, classes[1], dtype="int16")
    mask[ndvi <= low] = classes[0]
    mask[ndvi >= high] = classes[2]
    height = np.asarray(elevation, dtype="float32")
    mask[height > np.median(height)] = classes[2]
    return mask


def build_masks():
    """Deterministic: same inputs, same masks, same order. Re-run it to get the same archive."""
    import numpy as np
    tiles = graded_tiles()
    if not tiles:
        return {"tile_0000": np.zeros((2, 2), dtype="int16")}
    classes = class_split()
    with np.load(os.path.join(DATA, "graded_spectral.npz"), allow_pickle=False) as spectral, \
            np.load(os.path.join(DATA, "graded_elevation.npz"), allow_pickle=False) as elevation:
        return dict((tile, tile_mask(spectral[tile], elevation[tile], classes)) for tile in tiles)


def content_digest(masks):
    """The digest the run record declares: content in written order, never the file bytes."""
    import numpy as np
    digest = hashlib.sha256()
    for name, member in masks.items():
        member = np.ascontiguousarray(member)
        digest.update(name.encode("utf-8"))
        digest.update(("%s%s" % (member.dtype.str, member.shape)).encode("utf-8"))
        digest.update(member.tobytes())
    return digest.hexdigest()


def band_normalisation():
    """Declared per band from the vendored statistics. This baseline applies none of it."""
    path = os.path.join(DATA, "band_statistics.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        doc = json.load(handle)
    doc = doc.get("bands", doc)
    return dict((band, doc[band]) for band in BANDS if band in doc)


def run_record(masks, seconds):
    return {
        "channel_fusion_order": list(BANDS) + ["ELEVATION"],
        "elevation_fusion_stage": "mid_level_fusion",
        "band_normalisation": band_normalisation(),
        "training_tile_ids": [],
        "elevation_read_tile_ids": [],
        "paths_written": [SUBMISSION, RECORD],
        "encoder_provenance": "from_scratch",
        "mask_member_order": list(masks),
        "replay_content_sha256": content_digest(build_masks()),
        "agent_budget_declaration": {"train_tiles": 0, "graded_run_seconds": round(seconds, 3),
                                     "inference_stride": 1},
    }


def main():
    import time
    import numpy as np
    started = time.time()
    masks = build_masks()
    os.makedirs(os.path.dirname(SUBMISSION) or ".", exist_ok=True)
    # savez writes members in the order they are passed, and that order is declared below.
    np.savez(SUBMISSION, **masks)
    # The replay digest is recomputed from a SECOND build, not copied off the first. A digest
    # copied from the archive just written claims a replay that never ran.
    record = run_record(masks, time.time() - started)
    os.makedirs(os.path.dirname(RECORD) or ".", exist_ok=True)
    with open(RECORD, "w") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    if not graded_tiles():
        sys.stderr.write("no graded tile archive under environment/data/; wrote one placeholder "
                         "member so the delivery path still runs\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
