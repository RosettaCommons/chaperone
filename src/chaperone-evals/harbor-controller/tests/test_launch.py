from __future__ import annotations

import unittest

from launch import _gpu_selection


class GpuSelectionTests(unittest.TestCase):
    def test_explicit_count_selects_exact_devices(self) -> None:
        self.assertEqual(_gpu_selection("2", {}), (2, "--gpus", ["0", "1"]))

    def test_explicit_count_respects_existing_assignment(self) -> None:
        self.assertEqual(
            _gpu_selection("1", {"CUDA_VISIBLE_DEVICES": "3,7"}),
            (1, "--gpus constrained by CUDA_VISIBLE_DEVICES", ["3"]),
        )

    def test_explicit_count_rejects_too_few_assigned_devices(self) -> None:
        with self.assertRaisesRegex(Exception, "exceeds the 1 devices"):
            _gpu_selection("2", {"CUDA_VISIBLE_DEVICES": "3"})

    def test_auto_uses_visible_device_count(self) -> None:
        self.assertEqual(
            _gpu_selection("auto", {"CUDA_VISIBLE_DEVICES": "3,7"}),
            (2, "CUDA_VISIBLE_DEVICES", ["3", "7"]),
        )


if __name__ == "__main__":
    unittest.main()
