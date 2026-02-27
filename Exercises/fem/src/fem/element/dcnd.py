"""
Finite element conductive (heat transfer) elements (DCPn).

Analogous to cnd.py, but for thermal conduction.
Temperature DOF at index 3 of 10 in the node freedom table.
History variables track element-level quantities (e.g., flux),
not nodal temperature.
"""

from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from ..material import Material
from .geom import P3, P4
from .isop import IsoparametricElement


class DCnD:
    """
    Conductive element mixin.

    Attributes
    ----------
    ndof : int
        Number of temperature DOFs per node (1 for temperature).
    """

    ndof: int = 1

    def update_state(
        self,
        material: Material,
        step: int,
        increment: int,
        time: Sequence[float],
        dt: float,
        eleno: int,
        p: NDArray,
        T: NDArray,
        dT: NDArray,
        hsv: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Compute element conductivity and internal flux variables.

        Parameters
        ----------
        material : Material
            Material with thermal conductivity evaluation.
        step, increment : int
            Step and increment counters.
        time : sequence of float
            Current time values.
        dt : float
            Time increment.
        eleno : int
            Element number.
        p : NDArray
            Nodal coordinates.
        T : NDArray
            Current nodal temperatures.
        dT : NDArray
            Temperature increment (unused here).
        hsv : NDArray
            Element history variable array to store flux or energy.

        Returns
        -------
        K : NDArray
            Element conductivity matrix.
        q : NDArray
            Element internal flux vector.
        """
        K, q = material.eval(T)
        hsv[:] = q  # store element flux or internal quantities
        return K, q


class DCP3(P3, DCnD, IsoparametricElement):
    """3-node constant conductivity triangle element."""

    gauss_pts = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]], dtype=float) / 6.0
    gauss_wts = np.array([1.0, 1.0, 1.0], dtype=float) / 6.0
    edge_gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    edge_gauss_wts = np.array([1.0, 1.0], dtype=float)

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        """
        Node freedom table of length 10.
        Temperature DOF at literal index 3 (4th entry).
        """
        return [
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        ]

    def pmatrix(self, xi: NDArray) -> NDArray:
        """Interpolation matrix for temperature DOF."""
        N = self.shape(xi)
        P = np.zeros((1, 3))
        P[0, :] = N
        return P


class DCP4(P4, DCnD, IsoparametricElement):
    """4-node constant conductivity quadrilateral element."""

    gauss_pts = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]) / np.sqrt(3.0)
    gauss_wts = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    edge_gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    edge_gauss_wts = np.array([1.0, 1.0], dtype=float)

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        """Node freedom table of length 10, temperature at index 3."""
        return [
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        ]

    def pmatrix(self, xi: NDArray) -> NDArray:
        """Interpolation matrix for temperature DOF."""
        N = self.shape(xi)
        P = np.zeros((1, 4))
        P[0, :] = N
        return P


# Element-level quantities (history variables) for internal flux
class DCP3Flux(DCP3):
    def history_variables(self) -> list[str]:
        return ["q_x", "q_y"]


class DCP4Flux(DCP4):
    def history_variables(self) -> list[str]:
        return ["q_x", "q_y"]