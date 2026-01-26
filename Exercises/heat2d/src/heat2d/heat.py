import argparse
import sys
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from distmeshpy import distmesh2d
from distmeshpy.utils import dcircle
from distmeshpy.utils import ddiff
from distmeshpy.utils import drectangle
from distmeshpy.utils import huniform
from numpy.typing import NDArray

# Gauss quadrature points
tri_gauss_points = [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]]
tri_gauss_wts = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]

lin_gauss_points = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
lin_gauss_wts = [1.0, 1.0]

tri_edges = [[0, 1], [1, 2], [2, 0]]

NEUMANN = 0
DIRICHLET = 1
CONVECTION = 2
CONDUCTION = 3


def heat2d(
    p: NDArray[float],
    t: NDArray[float],
    D: NDArray[float],
    dbcs: list[tuple[int, float]] = [],
    nbcs: list[tuple[int, int, int, tuple[float, ...]]] = [],
    source: Callable[[float, float], float] = lambda x, y: 0.0,
) -> tuple[NDArray[float], NDArray[float]]:
    """Solve the 2D heat problem

    Args:
        p: nodal coordinates
        t: triangulation.  t[:, i] are the ordered node numbers of the ith element
        D: D[e] is the thermal conductivity of the eth element
        dbcs: Dirichlet BCs.  dbcs[i] = [int, float] is the
            node number and prescribed temperature of the ith BC
        nbcs: Neumann BCs.  nbcs[i] = [int, int, int, tuple] is the
            element number, edge number, BC type, and value of the ith BC
            The BC type should be one of CONDUCTION or CONVECTION
            If BC type is CONVECTION, then value = (h, Too) (convection coefficient and far field temp)
            If BC type is CONDUCTION, then value = (q,) (conduction on edge)
        source: Heat generation (s(x, y))

    """
    K = np.zeros((p.shape[0], p.shape[0]))
    F = np.zeros(p.shape[0])

    for e, nodes in enumerate(t):
        pe = p[nodes]
        Be = shapegrad(pe)
        Ae = area(pe)
        Je = 2.0 * Ae

        # Element stiffness matrix
        ke = Ae * D[e] * np.dot(Be.T, Be)

        # Element body source array
        # The equivalent nodal fluxes for the spatially-varying source is found by integrating
        # {Ne(x,y)} * s(x,y), over the element domain using 3-pt Gauss quadrature.
        fe = np.zeros(3)
        for w, xi in zip(tri_gauss_wts, tri_gauss_points):
            x = pe[0, 0] * xi[0] + pe[1, 0] * xi[1] + pe[2, 0] * (1.0 - xi[0] - xi[1])
            y = pe[0, 1] * xi[0] + pe[1, 1] * xi[1] + pe[2, 1] * (1.0 - xi[0] - xi[1])
            Ne = shape(pe, x, y)
            fe += Je * w * Ne * source(x, y)

        # Element boundary source array
        # The equivalent nodal fluxes for the Neumann BCs is found by integrating
        # {Ne(x,y)} * q, along the effected edge using 2-pt Gauss quadrature.
        for nbc in nbcs:
            eid, edge, type, value = nbc
            if eid != e:
                continue
            nft = tri_edges[edge]  # local node freedom table
            xa, xb = pe[nft, 0].min(), pe[nft, 0].max()
            ya, yb = pe[nft, 1].min(), pe[nft, 1].max()
            if type == CONVECTION:
                h, Too = value
                raise NotImplementedError("Complete this section")
            elif type == CONDUCTION:
                qb, *_ = value
                for w, xi in zip(lin_gauss_wts, lin_gauss_points):
                    x = 0.5 * (xb - xa) * xi + 0.5 * (xb + xa)
                    y = 0.5 * (yb - ya) * xi + 0.5 * (yb + ya)
                    le = np.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
                    J = le / 2.0
                    N = shape(pe, x, y)
                    fe[nft] += qb * N[nft] * w * J

        K[np.ix_(nodes, nodes)] += ke
        F[nodes] += fe

    # Enforce boundary conditions
    Kbc = K.copy()
    Fbc = F.copy()

    # Left boundary
    for node, T in dbcs:
        raise NotImplementedError("Complete this section")

    # Solve for nodal temperature and reactions
    temp = np.linalg.solve(Kbc, Fbc)
    r = np.dot(K, temp) - F

    return temp, r


def area(p: NDArray[float]) -> float:
    """Find the area of the triangle subtended by points p

    Args:
        p: p[:,0] are the x points, p[:,1] are the y points

    """
    xp, yp = p[:, 0], p[:, 1]
    d = xp[0] * (yp[1] - yp[2]) + xp[1] * (yp[2] - yp[0]) + xp[2] * (yp[0] - yp[1])
    return d / 2.0


def shape(p: NDArray[float], x: float, y: float) -> NDArray[float]:
    A = area(p)
    xp, yp = p[:, 0], p[:, 1]
    N = np.zeros(3)
    N[0] = xp[1] * yp[2] - xp[2] * yp[1] + (yp[1] - yp[2]) * x + (xp[2] - xp[1]) * y
    N[1] = xp[2] * yp[0] - xp[0] * yp[2] + (yp[2] - yp[0]) * x + (xp[0] - xp[2]) * y
    N[2] = xp[0] * yp[1] - xp[1] * yp[0] + (yp[0] - yp[1]) * x + (xp[1] - xp[0]) * y
    return N / 2.0 / A


def shapegrad(p: NDArray[float]) -> NDArray[float]:
    A = area(p)
    xp, yp = p[:, 0], p[:, 1]
    B = np.zeros((2, 3))
    B[0, :] = [yp[1] - yp[2], yp[2] - yp[0], yp[0] - yp[1]]
    B[1, :] = [xp[2] - xp[1], xp[0] - xp[2], xp[1] - xp[0]]
    return B / 2.0 / A


def tplot(p: NDArray[float], t: NDArray[int], temp: NDArray[float]) -> None:
    """Make temperature contour plot"""
    triang = tri.Triangulation(p[:, 0], p[:, 1], t)
    plt.figure(figsize=(7, 5))
    countour = plt.tricontourf(triang, temp, levels=50, cmap="turbo")
    plt.triplot(triang, color="k", linewidth=0.3)
    plt.colorbar(countour, label="Temperature")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Temperature field")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.cla()
    plt.close("all")


def rplot(p: NDArray[float], t: NDArray[int], r: NDArray[float]) -> None:
    """Make plots of reactions on left and right edges"""
    _, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Left reaction
    ilo = [
        n for n, x in enumerate(p) if isclose(x[0], -1.0)
    ]  # Left boundary:  node numbers for x = -1
    ylo = p[ilo, 1]
    rlo = r[ilo]
    ix = np.argsort(ylo)
    ylo = ylo[ix]
    rlo = rlo[ix]

    axs[0].plot(ylo, rlo, "o", label="LHS")
    axs[0].set_title("LHS reaction")
    axs[0].set_xlabel("y")
    axs[0].set_ylabel("Heat flux/reaction")
    axs[0].grid(True)

    # Right reaction
    ihi = [
        n for n, x in enumerate(p) if isclose(x[0], 1.0)
    ]  # Right boundary:  node numbers for x = 1
    yhi = p[ihi, 1]
    rhi = r[ihi]
    ix = np.argsort(yhi)
    yhi = yhi[ix]
    rhi = rhi[ix]

    axs[1].plot(yhi, rhi, "o", label="RHS")
    axs[1].set_title("RHS reaction")
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("Heat flux/reaction")
    axs[1].grid(True)

    print(np.sum(rlo))
    print(np.sum(rhi))

    plt.tight_layout()
    plt.show()


def indices(a: list[int], values: list[int]) -> list[int]:
    return [i for i, n in enumerate(a) if n in values]


def isclose(a, b, rtol: float = 0.0001, atol: float = 1e-8) -> bool:
    return abs(a - b) <= (atol + rtol * abs(b))


def example(esize: float = 0.05):
    """
    Solve the 2D heat equation over a square domain

    • Bounds: x ∈ [-1, 1], y ∈ [-1, 1] with hole of radius .3 at its center.
    • Spatially varying heat source: 1000 / √(x^2 + y^2)
    • Fixed temperature on left edge: 200˚
    • Fixed temperature on right edge: 50˚
    • Heat flux along bottom edge: 2000
    • Convection along top edge: far field temperature 25˚ with convection coefficient 250

    """
    fd = lambda p: ddiff(drectangle(p, -1, 1, -1, 1), dcircle(p, 0, 0, 0.5))
    fh = lambda p: 0.05 + 0.3 * dcircle(p, 0, 0, 0.5)
    bbox = ((-1.0, 1.0), (-1.0, 1.0))
    fixed = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    p, t = distmesh2d(fd, fh, esize, bbox, fixed)

    # Generate Dirichlet BCs: fixed temperature on left (200˚) and right (50˚) sides
    dbcs: list[tuple[int, float]] = []
    for node in np.where(np.abs(p[:, 0] + 1.0) < 1e-6)[0]:
        dbcs.append((node, 200.0))
    for node in np.where(np.abs(p[:, 0] - 1.0) < 1e-6)[0]:
        dbcs.append((node, 50.0))

    # Generate Neumann BCs: convection on top surface and conduction on bottom
    nbcs: list[tuple[int, int, int, tuple[float, ...]]] = []
    for e, nodes in enumerate(t):
        jhi = [node for node in nodes if isclose(p[node, 1], 1.0)]
        if len(jhi) == 2:
            li, lj = indices(nodes, jhi)
            for edge, (a, b) in enumerate(tri_edges):
                if {a, b} == {li, lj}:
                    nbcs.append((e, edge, CONVECTION, (250.0, 25.0)))
                    break
            continue
        jlo = [node for node in nodes if isclose(p[node, 1], -1.0)]
        if len(jlo) == 2:
            li, lj = indices(nodes, jlo)
            for edge, (a, b) in enumerate(tri_edges):
                if {a, b} == {li, lj}:
                    nbcs.append((e, edge, CONDUCTION, (2000.0,)))
                    break
            continue

    # Heat source
    s = lambda x, y: 1000.0 / np.sqrt(x**2 + y**2)
    D = np.ones(t.shape[0]) * 12.0
    temp, r = heat2d(p, t, D, source=s, nbcs=nbcs, dbcs=dbcs)
    tplot(p, t, temp)
    rplot(p, t, r)


def verify(esize: float = 0.05):
    """
    Solve the 2D heat equation over a square domain

    • Bounds: x ∈ [-1, 1], y ∈ [-1, 1]
    • Spatially varying heat source: None
    • Left and right edges insulated.
    • Heat flux along bottom edge: 2000
    • Convection along top edge: far field temperature 25˚ with convection coefficient 250

    Analytic solution:

    T = 33˚ along the bottom edge
    T = 366.333˚ along the top edge

    """
    fd = lambda p: drectangle(p, -1, 1, -1, 1)
    fh = lambda p: huniform(p)
    bbox = ((-1.0, 1.0), (-1.0, 1.0))
    fixed = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    p, t = distmesh2d(fd, fh, esize, bbox, fixed)

    # Generate Neumann BCs: convection on top surface and conduction on bottom
    nbcs: list[tuple[int, int, int, tuple[float, ...]]] = []
    for e, nodes in enumerate(t):
        jhi = [node for node in nodes if isclose(p[node, 1], 1.0)]
        if len(jhi) == 2:
            li, lj = indices(nodes, jhi)
            for edge, (a, b) in enumerate(tri_edges):
                if {a, b} == {li, lj}:
                    nbcs.append((e, edge, CONVECTION, (250.0, 25.0)))
                    break
            continue
        jlo = [node for node in nodes if isclose(p[node, 1], -1.0)]
        if len(jlo) == 2:
            li, lj = indices(nodes, jlo)
            for edge, (a, b) in enumerate(tri_edges):
                if {a, b} == {li, lj}:
                    nbcs.append((e, edge, CONDUCTION, (2000.0,)))
                    break
            continue

    # No heat source
    s = lambda x, y: 0.0
    D = np.ones(t.shape[0]) * 12.0
    temp, r = heat2d(p, t, D, source=s, nbcs=nbcs)
    print(temp[np.where(np.abs(p[:, 1] - 1.0) < 1e-6)[0]])
    print(temp[np.where(np.abs(p[:, 1] + 1.0) < 1e-6)[0]])
    tplot(p, t, temp)


def mms(esize: float = 0.05):
    raise NotImplementedError


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?", default="example", choices=("verify", "mms", "example"))
    p.add_argument("-s", type=float, default=0.05, help="Element size [default: %(default)s]")
    args = p.parse_args()
    if args.problem == "example":
        example(esize=args.s)
    elif args.problem == "verify":
        verify(esize=args.s)
    elif args.problem == "mms":
        mms(esize=args.s)
    else:
        p.error(f"Unknown problem {args.problem}")


if __name__ == "__main__":
    sys.exit(main())
