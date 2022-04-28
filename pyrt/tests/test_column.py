import numpy as np
import pytest
from pyrt import Column, decompose_hg


class TestColumn:
    @pytest.fixture
    def non_numeric_angles(self) -> float:
        yield np.array([0, 10, {'a': 1}])

    def test_negative_optical_depth_raises_value_error(self):
        od = np.linspace(-0.1, 1, num=15)
        ssa = np.ones((15,))
        pmom = np.broadcast_to(decompose_hg(0, 128), (15, 128)).T

        with pytest.raises(ValueError):
            Column(od, ssa, pmom)

    def test_od_gives_expected_result(self):
        od0 = [1]
        od1 = [2]
        ssa = [1]
        pmom = np.array([[0, 1, 2]]).T
        col0 = Column(od0, ssa, pmom)
        col1 = Column(od1, ssa, pmom)
        col = col0 + col1
        assert col.optical_depth == 3

    def test_ssa_gives_expected_result(self):
        od = [1]
        ssa0 = [1]
        ssa1 = [0.5]
        pmom = np.array([[0, 1, 2]]).T
        col0 = Column(od, ssa0, pmom)
        col1 = Column(od, ssa1, pmom)
        col = col0 + col1
        assert col.single_scattering_albedo == 0.75
