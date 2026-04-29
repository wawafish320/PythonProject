from __future__ import annotations

import unittest

from tools import run_strict_replace_phasez_boundary_probe as _probe


class StrictReplacePhaseZBoundaryProbeTest(unittest.TestCase):
    def test_phase_z_carrier_span_replace_contacts_uses_trailing_phase_cols(self) -> None:
        self.assertEqual(
            _probe._phase_z_carrier_column_span(
                in_features=43,
                contact_dim=2,
                use_phase_z=True,
                phase_z_mode="replace_contacts",
            ),
            (39, 43),
        )

    def test_phase_z_carrier_span_concat_uses_trailing_phase_cols(self) -> None:
        self.assertEqual(
            _probe._phase_z_carrier_column_span(
                in_features=47,
                contact_dim=2,
                use_phase_z=True,
                phase_z_mode="concat",
            ),
            (43, 47),
        )

    def test_phase_z_carrier_span_requires_phase_z(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "direct_pose_use_phase_z=true"):
            _probe._phase_z_carrier_column_span(
                in_features=43,
                contact_dim=2,
                use_phase_z=False,
                phase_z_mode="replace_contacts",
            )


if __name__ == "__main__":
    unittest.main()
