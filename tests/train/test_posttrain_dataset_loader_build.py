from __future__ import annotations

from types import SimpleNamespace
import unittest

from train import posttrain


class _ArrayStub:
    def __init__(self, length: int) -> None:
        self.shape = (length,)


class _SizedStub:
    def __init__(self, length: int, *, clips=None) -> None:
        self._length = int(length)
        self.clips = list(clips or [])

    def __len__(self) -> int:
        return self._length


class PosttrainDatasetLoaderBuildTest(unittest.TestCase):
    def test_collect_posttrain_clip_lengths_sorts_and_skips_bad_clips(self) -> None:
        ds = _SizedStub(
            0,
            clips=[
                SimpleNamespace(npz_path="/tmp/long_clip.npz", X=_ArrayStub(20)),
                SimpleNamespace(npz_path="/tmp/short_clip.npz", X=_ArrayStub(3)),
                SimpleNamespace(npz_path="/tmp/bad_clip.npz", X=object()),
            ],
        )

        self.assertEqual(
            posttrain._collect_posttrain_clip_lengths(ds),
            [
                ("/tmp/short_clip.npz", 3),
                ("/tmp/long_clip.npz", 20),
            ],
        )

    def test_assert_posttrain_dataset_has_samples_includes_smallest_clip_hint(self) -> None:
        ds = _SizedStub(
            0,
            clips=[
                SimpleNamespace(npz_path="/tmp/clip_b.npz", X=_ArrayStub(12)),
                SimpleNamespace(npz_path="/tmp/clip_a.npz", X=_ArrayStub(5)),
            ],
        )

        with self.assertRaises(SystemExit) as cm:
            posttrain._assert_posttrain_dataset_has_samples(ds=ds, seq_len=87)

        msg = str(cm.exception)
        self.assertIn("posttrain dataset has 0 samples", msg)
        self.assertIn("seq_len=87", msg)
        self.assertIn("Smallest clips: clip_a.npz:5, clip_b.npz:12.", msg)

    def test_assert_posttrain_loader_has_batches_reports_dataset_and_batch(self) -> None:
        loader = _SizedStub(0)
        ds = _SizedStub(1)

        with self.assertRaises(SystemExit) as cm:
            posttrain._assert_posttrain_loader_has_batches(loader=loader, ds=ds, batch=4)

        msg = str(cm.exception)
        self.assertIn("posttrain DataLoader has 0 batches", msg)
        self.assertIn("len(dataset)=1", msg)
        self.assertIn("batch=4", msg)


if __name__ == "__main__":
    unittest.main()
