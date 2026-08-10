from __future__ import annotations

import unittest

from PIL import Image

from evaluate_coco_text import annotation_mask


class CocoTextEvaluationTests(unittest.TestCase):
    def test_annotation_mask_uses_only_selected_polygon(self) -> None:
        mask = annotation_mask(
            (20, 20),
            {
                "polygon": [[2, 3], [8, 3], [8, 9], [2, 9]],
                "bbox": [2, 3, 6, 6],
            },
        )
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.getpixel((4, 5)), 255)
        self.assertEqual(mask.getpixel((15, 15)), 0)

    def test_annotation_mask_falls_back_to_bbox(self) -> None:
        mask = annotation_mask((20, 20), {"bbox": [3, 4, 5, 6]})
        self.assertEqual(mask.getbbox(), (3, 4, 9, 11))


if __name__ == "__main__":
    unittest.main()
