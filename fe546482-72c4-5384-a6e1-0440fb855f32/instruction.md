# AELLO-C8-S6

Maximise box-level localisation accuracy on a private split of Oxford-IIIT Pet when only a fixed small fraction of the training images carries a box.

## What is graded

Write your graded artifact to **`/submission/detections.json`**. A missing, malformed or out-of-domain artifact scores 0.

```json
{
  "detections": {
    "Abyssinian_101": [
      {"box": [34.0, 12.0, 291.0, 268.0], "score": 0.93},
      {"box": [8.0, 3.0, 380.0, 330.0], "score": 0.41}
    ],
    "beagle_17": [{"box": [61.0, 88.0, 402.0, 375.0], "score": 0.77}]
  }
}
```

- One key per **graded image id**, and every graded image must have one. The list may not be empty.
- `box` is `[x1, y1, x2, y2]` in **absolute pixels on that image's own native frame**, x to the right and y down, half-open: width is `x2 - x1`. Oxford-IIIT Pet images are not all one size; `environment/data/graded_images.json` publishes each image's width and height and your box is checked against **that** image's numbers. A box left in the 0-to-1 normalised frame, or on the fixed square your network was fed, is out of domain.
- `score` is a finite number. Within one image's list, **exactly one box may hold the top score.**

Your score is `r_det`, computed by the verifier in a separate environment from state it measures itself.

## The graded quantity: correct-localisation at IoU 0.5

For each image in the private split, your **highest-scoring box** is intersected with that image's single ground-truth box. The image counts if the intersection over union is **at least 0.5**. The rate is that count over **the size of the private split**.

Three consequences worth reading twice:

- **Only the top box is read.** Every other box in a list is inert -- it cannot raise the rate and it cannot lower it. This is not mean average precision: there is no ranked list to tune, and widening recall buys nothing.
- **The denominator is the split, not your keys.** An image you do not answer is a miss. Under mAP, withholding an uncertain box protects precision; here it is strictly worse than guessing, because a guess can still land at 0.5.
- **The threshold is a cliff, not a slope.** An IoU of 0.4999 scores exactly what an empty box scores, and 0.9 scores exactly what 0.5 scores. Tightening a box that is already over the line is worth nothing; the only images that pay are the ones sitting just under it.

`r_det` is zero on any red line and on any knockout failure, before any quality is considered.

## The supervision you have

- `environment/data/training_images.json` -- every training image id with its width and height.
- `environment/data/labelled_boxes.json` -- **the fixed small fraction**: image id to `[x1, y1, x2, y2]`. This is the only box supervision that exists. It cannot be enlarged, and reading a box for any image it does not name is a failure of `box_supervision_confined_to_the_labelled_fraction`.
- `environment/data/graded_images.json` -- the graded image ids with their native width and height. You may read these pixels; you must predict on them.

The graded **boxes** live under `tests/heldout/` and the graded run must not read them. The per-pixel **trimaps** Oxford-IIIT Pet ships are withheld from this tree entirely -- thresholding one gives a tight box on every training image, which is the dense supervision this task is defined by the absence of. Their absence is a compiled check, in the tree and in your read declaration.

## The run record

Write `/workspace/run_record.json` during the graded run. The verifier reads it and the compiled checks below grade it, so an absent or incomplete record costs score on its own:

| key | what it must carry |
| --- | --- |
| `box_supervised_image_ids` | every image id whose ground-truth box you read. Must be a subset of `labelled_boxes.json`, and must contain no graded image id |
| `labelled_fraction` | the supervision fraction you were given: the size of `labelled_boxes.json` over the size of `training_images.json` |
| `pseudo_box_image_ids` | every image you placed a propagated pseudo-box on; disjoint from `labelled_boxes.json`, since an image that ships a box needs no pseudo-box |
| `pseudo_box_accounting` | `{"proposed": int, "retained": int, "confidence_gate": float}` -- the gate strictly between 0 and 1, `retained` no greater than `proposed`. A gate that retained every proposal admitted everything, and that is trap two |
| `paths_read` | every path the graded run opened |
| `paths_written` | every path the graded run wrote |
| `replay_top_box_sha256` | the digest of the decision your seeded replay reproduces (below) |
| `agent_budget_declaration` | `{"labelled_images_used": int, "graded_run_seconds": number, "inference_scales": [numbers], "optimised_for_iou": 0.5}` -- scales distinct and positive, one entry if inference was single-scale; `labelled_images_used` no greater than the fraction ships |

If you ran no propagation, declare it: `pseudo_box_image_ids: []` and `pseudo_box_accounting: {"proposed": 0, "retained": 0, "confidence_gate": <your gate>}`.

### `replay_top_box_sha256`

A sha256 over what your run **decided**, not over the file it wrote. For each image id in your submission, in sorted order, feed one UTF-8 line:

```
<image_id>|<x1>,<y1>,<x2>,<y2>\n      # each coordinate of the top-scoring box, floored to its integer pixel
<image_id>|none\n                     # where the list is empty or two boxes share the top score
```

The digest is over the decision because reordering the inert boxes below the top one, or printing a score as `0.30000000000000004` on one accumulation order, changes no localisation and must not read as divergence -- while a top box that moves by one pixel **can** cross IoU 0.5 and flip a graded image, so it must.

## The budget

- One graded run. `max_timeout` bounds the refinement loop; each attempt must finish inside its per-attempt budget.
- Up to 50 attempts. Your reward history is carried between attempts.
- You must declare your own budget accounting for the graded run; the declaration is what is graded, not the harness's own measurement.

## Red lines

Any of these scores zero:

- Any read of the private split's boxes -- the ids in `box_supervised_image_ids` may not intersect the graded split, and `paths_read` may not name `tests/heldout/`.
- Any write into `tests/` during the graded run, or onto `graded_boxes.json`, `labelled_boxes.json` or `graded_images.json`.

## What ships

- `environment/data/` -- the training images, the graded images, and the fixed small fraction of boxes.
- `environment/starter/` -- a correct, unoptimised baseline you may replace entirely. It answers every graded image with one full-frame box and writes a complete run record, so the delivery path is exercised before you change a line of it. It is deliberately weak on the metric: a whole-image box is not nothing on this corpus, and beating it is the job.

## Traps

- Treats the unlabelled majority as background. The detector learns that most images contain nothing, emits an empty list per image, and the rate is zero -- not because the localisation was poor but because none was attempted.
- Propagates pseudo-boxes with no confidence gate. Every accepted error becomes supervision for the next round and the training set degrades faster than the extra volume helps.
- Submits a ranked list per image and tunes it for recall. Only the top box is read; the rest is effort spent outside the metric.
- Withholds a box on images it is unsure about. The denominator is the whole graded split, so silence is a miss and a guess is strictly better.
- Optimises overlap it already has, as a recipe tuned for the averaged 0.50-to-0.95 sweep would. At a single hard threshold that is worth zero.
- Returns boxes in the normalised frame or on the fixed square the network was fed, instead of each image's own native frame.
- Gives every box the same score, so the top box is whichever one the serialiser wrote first.
- Reaches for the trimaps. They are withheld from the delivered tree and their absence is a compiled check.
