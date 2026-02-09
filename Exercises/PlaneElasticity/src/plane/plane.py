import argparse
import sys
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from distmeshpy import distmesh2d
from distmeshpy.utils import dcircle
from distmeshpy.utils import ddiff
from distmeshpy.utils import drectangle
from distmeshpy.utils import huniform
from numpy.typing import NDArray

np.set_printoptions(precision=2)

ArrayLike = NDArray | Sequence

X, Y = 0, 1
NEUMANN = 0
DIRICHLET = 1
ROBIN = 2
TRACTION = 3


@dataclass
class Mesh:
    coords: NDArray
    connect: NDArray
    nodesets: dict[str, "Nodeset"] = field(default_factory=dict)
    sidesets: dict[str, "Sideset"] = field(default_factory=dict)
    dof_per_node: int = field(default=2)
    nnode: int = field(init=False, default=-1)
    nelem: int = field(init=False, default=-1)
    node_per_elem: int = field(init=False, default=-1)

    def __post_init__(self) -> None:
        self.nnode = self.coords.shape[0]
        self.nelem = self.connect.shape[0]
        self.node_per_elem = self.connect.shape[1]


class Material:
    def __init__(self, density: float = 1.0, **properties: float) -> None:
        raise NotImplementedError

    def eval(self, ndir: int, nshr: int) -> NDArray:
        raise NotImplementedError


class LinearElastic(Material):
    def __init__(
        self, *, youngs_modulus: float, poissons_ratio: float, density: float = 1.0
    ) -> None:
        self.density = density
        self.youngs_modulus = youngs_modulus
        assert self.youngs_modulus > 0
        self.poissons_ratio = poissons_ratio
        assert -1 <= self.poissons_ratio < 0.5

    def eval(self, ndir: int, nshr: int) -> NDArray:
        E = self.youngs_modulus
        nu = self.poissons_ratio
        if ndir == 2 and nshr == 1:
            # Plane stress: 2 direct components of stress and 1 shear component
            return E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        elif ndir == 3 and nshr == 1:
            # Plane strain: 3 direct components of stress and 1 shear component
            factor = E / (1 - nu) / (1 - 2 * nu)
            return factor * np.array(
                [
                    [1 - nu, nu, nu, 0],
                    [nu, 1 - nu, nu, 0],
                    [nu, nu, 1 - nu, 0],
                    [0, 0, 0, (1 - 2 * nu) / 2],
                ]
            )
        raise NotImplementedError(f"{ndir=}, {nshr=}")


class Element(ABC):
    dof_per_node: int
    ndir: int
    nshr: int
    edges: NDArray
    ref_coords: NDArray
    gauss_pts: NDArray
    gauss_wts: NDArray
    edge_gauss_pts: NDArray
    edge_gauss_wts: NDArray

    def __init__(self, *, material: Material) -> None:
        self.material = material

    @abstractmethod
    def area(self, p: NDArray) -> float:
        """Find the area of the triangle subtended by points p

        Args:
            p: p[:,0] are the x points, p[:,1] are the y points

        """
        ...

    @abstractmethod
    def shape(self, xi: NDArray) -> NDArray: ...

    @abstractmethod
    def shapegrad(self, xi: NDArray) -> NDArray: ...

    @abstractmethod
    def pmatrix(self, xi: NDArray) -> NDArray: ...

    @abstractmethod
    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray: ...

    @abstractmethod
    def interpolate(self, p: NDArray, xi: NDArray) -> tuple[float, float]: ...

    @abstractmethod
    def jacobian(self, p: NDArray, xi: NDArray) -> float: ...

    @abstractmethod
    def interpolate_edge(self, p: NDArray, xi: float) -> tuple[float, float]: ...

    @abstractmethod
    def edge_jacobian(self, edge: int, p: NDArray, xi: float) -> float: ...

    @abstractmethod
    def edge_ref_coords(self, edge: int, xi: float) -> NDArray: ...


class CPX3(Element):
    dof_per_node: int = 2
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    ref_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    gauss_pts = np.array([[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]])
    gauss_wts = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    edge_gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    edge_gauss_wts = np.array([1.0, 1.0])

    def area(self, p: NDArray) -> float:
        xp, yp = p[:, 0], p[:, 1]
        d = xp[0] * (yp[1] - yp[2]) + xp[1] * (yp[2] - yp[0]) + xp[2] * (yp[0] - yp[1])
        assert d > 0
        return d / 2.0

    def shape(self, xi: NDArray) -> NDArray:
        s, t = xi
        N = np.array([1.0 - s - t, s, t])
        return N

    def pmatrix(self, xi: NDArray) -> NDArray:
        N = self.shape(xi)
        P = np.zeros((6, 2))
        P[0::2, 0] = N
        P[1::2, 1] = N
        return P

    def shapegrad(self, xi: NDArray) -> NDArray:
        return np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])

    def interpolate(self, p: NDArray, xi: NDArray) -> tuple[float, float]:
        N = self.shape(xi)
        x = np.dot(N, p[:, 0])
        y = np.dot(N, p[:, 0])
        return x, y

    def jacobian(self, p: NDArray, xi: NDArray) -> float:
        dNdxi = self.shapegrad(xi)
        dxdxi = np.dot(dNdxi, p)
        return np.linalg.det(dxdxi)

    def edge_ref_coords(self, edge: int, xi: float) -> NDArray:
        """
        Map 1D Gauss points on the edge to reference element (s,t).

        Returns:
            s, t: array of shape (2,) with reference coordinates for xi
        """
        a, b = self.edges[edge]
        sa, ta = self.ref_coords[a]
        sb, tb = self.ref_coords[b]
        s = 0.5 * (1 - xi) * sa + 0.5 * (1 + xi) * sb
        t = 0.5 * (1 - xi) * ta + 0.5 * (1 + xi) * tb
        return np.array((s, t))

    def edge_jacobian(self, edge: int, p: NDArray, xi: float) -> float:
        """
        Compute Jacobian |dx/dxi| for each 1D Gauss point along a physical edge.

        Args:
            p_edge: array of shape (n_edge_nodes, 2) with coordinates of edge nodes
        Returns:
            J: array of length ngauss with Jacobian at each Gauss point
        """
        # Map xi -> s, t
        st = self.edge_ref_coords(edge, xi)
        dNdst = self.shapegrad(st)
        dxdst = np.dot(dNdst, p)
        a, b = self.edges[edge]
        sa, ta = self.ref_coords[a]
        sb, tb = self.ref_coords[b]
        dstdxi = 0.5 * np.array([sb - sa, tb - ta])
        dxdxi = np.dot(dxdst.T, dstdxi)
        return np.linalg.norm(dxdxi)

    def interpolate_edge(self, p: NDArray, xi: float) -> tuple[float, float]:
        xp, yp = p[:, 0], p[:, 1]
        x = 0.5 * (1.0 - xi) * xp[0] + 0.5 * (1 + xi) * xp[1]
        y = 0.5 * (1.0 - xi) * yp[0] + 0.5 * (1 + xi) * yp[1]
        return x, y


class CPS3(CPX3):
    """Plane stress CST element"""

    ndir: int = 2
    nshr: int = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdxi = self.shapegrad(xi)
        dxdxi = np.dot(dNdxi, p)
        dNdx = np.dot(np.linalg.inv(dxdxi), dNdxi)
        B = np.zeros((3, 6))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[2, 0::2] = dNdx[1]
        B[2, 1::2] = dNdx[0]
        return B


class CPE3(CPX3):
    """Plane strain CST element"""

    ndir: int = 3
    nshr: int = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdxi = self.shapegrad(xi)
        dxdxi = np.dot(dNdxi, p)
        dNdx = np.dot(np.linalg.inv(dxdxi), dNdxi)
        B = np.zeros((4, 6))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[3, 0::2] = dNdx[1]
        B[3, 1::2] = dNdx[0]
        return B


@dataclass
class DirichletBC:
    nodeset: str
    dof: int
    name: str = "Dirichlet BC"
    value: float = 0.0


@dataclass
class Nodeset:
    name: str
    nodes: NDArray


@dataclass
class NeumannBC:
    sideset: str
    type: int
    value: NDArray


@dataclass
class Sideset:
    name: str
    sides: NDArray


class Model:
    def __init__(
        self,
        mesh: Mesh,
        elements: list[Element],
        neumann_bcs: list[NeumannBC] | None = None,
        dirichlet_bcs: list[DirichletBC] | None = None,
        source: Callable[[float, float], NDArray] | None = None,
    ) -> None:
        self.mesh = mesh
        self.elements = elements
        self.nnode = self.mesh.nnode
        self.nelem = self.mesh.nelem

        self.neumann_bcs: list[NeumannBC] = neumann_bcs or []
        self.dirichlet_bcs: list[DirichletBC] = dirichlet_bcs or []

        self.source: Callable[[float, float], NDArray] = source or (lambda x, y: np.zeros(2))

        # Derived properties
        self.dirichlet_dofs: NDArray = np.array([], dtype=int)
        self.dirichlet_vals: NDArray = np.array([], dtype=float)

        self.prepared = False

    def prepare(self) -> None:
        assert len(self.elements) == self.mesh.nelem
        m = self.mesh
        assert all([self.elements[e].area(m.coords[m.connect[e]]) > 0.0 for e in range(m.nelem)])
        ddofs: dict[int, float] = {}
        for dbc in self.dirichlet_bcs:
            ns = self.mesh.nodesets[dbc.nodeset]
            for node in ns.nodes:
                dof = int(node * self.mesh.dof_per_node + dbc.dof)
                ddofs[dof] = dbc.value
        self.dirichlet_dofs = np.array(list(ddofs.keys()), dtype=int)
        self.dirichlet_vals = np.array(list(ddofs.values()), dtype=float)
        self.prepared = True

    def dofmap(self, nodes: ArrayLike) -> list[int]:
        dof_per_node = self.mesh.dof_per_node
        return [dof_per_node * n + d for n in nodes for d in range(dof_per_node)]

    def assemble(self) -> tuple[NDArray, NDArray]:
        ndof = self.mesh.dof_per_node * self.nnode
        dof_per_elem = self.mesh.node_per_elem * self.mesh.dof_per_node
        K = np.zeros((ndof, ndof))
        Fvol = np.zeros(ndof)
        for e, nodes in enumerate(self.mesh.connect):
            el = self.elements[e]
            pe = self.mesh.coords[nodes]

            # Element body source array
            # The equivalent nodal fluxes for the spatially-varying source is found by integrating
            # {Ne(x,y)} * s(x,y), over the element domain using 3-pt Gauss quadrature.
            fe = np.zeros(dof_per_elem)
            ke = np.zeros((dof_per_elem, dof_per_elem))
            for w, xi in zip(el.gauss_wts, el.gauss_pts):
                x, y = el.interpolate(pe, xi)
                P = el.pmatrix(xi)
                B = el.bmatrix(pe, xi)
                J = el.jacobian(pe, xi)
                D = el.material.eval(ndir=el.ndir, nshr=el.nshr)
                ke += w * J * np.dot(np.dot(B.T, D), B)
                fe += w * J * np.dot(P, self.source(x, y))
            nft = self.dofmap(nodes)
            K[np.ix_(nft, nft)] += ke
            Fvol[nft] += fe
        return K, Fvol

    def robin_stiffness(self, K: NDArray) -> None:
        dof_per_elem = self.mesh.node_per_elem * self.mesh.dof_per_node
        for nbc in self.neumann_bcs:
            if nbc.type != ROBIN:
                continue
            ss = self.mesh.sidesets[nbc.sideset]
            # Element boundary source arrays
            ke = np.zeros((dof_per_elem, dof_per_elem))
            for e, edge in ss.sides:
                el: Element = self.elements[e]
                nodes = self.mesh.connect[e]
                pe = self.mesh.coords[nodes]
                edge_nodes = el.edges[edge]  # local node freedom table
                nft = self.dofmap(edge_nodes)
                H, _ = nbc.value
                for w, xi in zip(el.edge_gauss_wts, el.edge_gauss_pts):
                    st = el.edge_ref_coords(edge, xi)
                    P = el.pmatrix(st)[nft]
                    J = el.edge_jacobian(edge, pe, xi)
                    ke[np.ix_(nft, nft)] += w * J * np.dot(np.dot(P, H), P.T)
                NFT = self.dofmap(nodes)
                K[np.ix_(NFT, NFT)] += ke

    def external_force(self) -> NDArray:
        ndof = self.mesh.dof_per_node * self.nnode
        F = np.zeros(ndof)
        dof_per_elem = self.mesh.node_per_elem * self.mesh.dof_per_node
        for nbc in self.neumann_bcs:
            ss = self.mesh.sidesets[nbc.sideset]
            # Element boundary source arrays
            # The equivalent nodal fluxes for the Neumann BCs is found by integrating
            # {Ne(x,y)} * q, along the effected edge using 2-pt Gauss quadrature.
            for e, edge in ss.sides:
                fe = np.zeros(dof_per_elem)
                el: Element = self.elements[e]
                nodes = list(self.mesh.connect[e])
                pe = self.mesh.coords[nodes]
                edge_nodes = el.edges[edge]  # local node freedom table
                nft = self.dofmap(edge_nodes)
                if nbc.type == ROBIN:
                    H, t0 = nbc.value
                    for w, xi in zip(el.edge_gauss_wts, el.edge_gauss_pts):
                        st = el.edge_ref_coords(edge, xi)
                        P = el.pmatrix(st)[nft]
                        J = el.edge_jacobian(edge, pe, xi)
                        fe[nft] += w * J * np.dot(P, np.dot(H, t0))
                elif nbc.type == TRACTION:
                    tx, ty = nbc.value
                    for w, xi in zip(el.edge_gauss_wts, el.edge_gauss_pts):
                        st = el.edge_ref_coords(edge, xi)
                        P = el.pmatrix(st)[nft]
                        J = el.edge_jacobian(edge, pe, xi)
                        fe[nft] += w * J * np.dot(P, [tx, ty])
                NFT = self.dofmap(nodes)
                F[NFT] += fe
        return F

    def solve(self) -> tuple[NDArray, NDArray]:
        """Solve the 2D plane stress/strain problem

        Args:
            mesh: Mesh
            elements: Elements
            dbcs: Dirichlet BCs.  dbcs[i] = [int, int, float] is the
                node number, local degree of freedom, and prescribed displacment of the ith BC
                0 based indexing.  local degrees of freedom range from 0 to number of dimensions - 1
            nbcs: Neumann BCs.  nbcs[i] = [int, int, int, tuple] is the
                element number, edge number, BC type, and value of the ith BC
                The BC type should be one of TRACTION or ROBIN
                If BC type is ROBIN, then value = (H, t0) where H is a 2x2 matrix
                If BC type is TRACTION, then value = (tx, ty)
            source: Body force.  b = s(x, y) is an array (bx, by) at the point (x, y)

        Returns:
            d: nodal displacements returned with shape (nnode, 2)
            r: nodal reactions (including Neumann terms) returned with shape (nnode, 2)

        """
        if not self.prepared:
            self.prepare()
        K, Fvol = self.assemble()
        self.robin_stiffness(K)
        Fext = self.external_force()
        F: ArrayLike = Fvol + Fext
        ddofs, dvals = self.dirichlet_dofs, self.dirichlet_vals
        fdofs = sorted(set(range(K.shape[0])).difference(ddofs))
        d = np.zeros(K.shape[0])
        d[ddofs] = dvals
        d[fdofs] = np.linalg.solve(
            K[np.ix_(fdofs, fdofs)], F[fdofs] - np.dot(K[np.ix_(fdofs, ddofs)], d[ddofs])
        )
        r = np.dot(K, d) - F

        return d.reshape((self.nnode, -1)), r.reshape((self.nnode, -1))


def tplot(p: NDArray, t: NDArray, z: NDArray) -> None:
    """Make a 2D contour plot

    Args:
      p: mesh point coordinates (n, 2)
      t: mesh connectivity (triangulation) (N, 3)
      z: array of points to plot (n)

    """
    triang = tri.Triangulation(p[:, 0], p[:, 1], t)
    plt.figure(figsize=(7, 5))
    countour = plt.tricontourf(triang, z, levels=50, cmap="turbo")
    plt.triplot(triang, color="k", linewidth=0.3)
    plt.colorbar(countour, label=None)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Temperature field")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.cla()
    plt.close("all")


def indices(a: Sequence[int], values: Sequence[int]) -> list[int]:
    return [i for i, n in enumerate(a) if n in values]


def find_edge(element: Element, conn: Sequence[int], nodes: Sequence[int]) -> int:
    """Given ordered nodes ``conn`` that define a triangles connectivity, find the edge defined by
    edge nodes ``nodes``"""
    li, lj = indices(conn, nodes)
    for edge, (a, b) in enumerate(element.edges):
        if {a, b} == {li, lj}:
            return edge
    raise ValueError(f"Unable to determine edge for nodes {nodes} in triangle {conn}")


def isclose(a, b, rtol: float = 0.0001, atol: float = 1e-8) -> bool:
    return abs(a - b) <= (atol + rtol * abs(b))


def exercise(esize: float = 0.05):
    p, t = plate_with_hole(esize=esize)
    mesh = Mesh(coords=p, connect=t)

    m = LinearElastic(density=2400.0, youngs_modulus=30.0e9, poissons_ratio=0.3)
    elements = [CPS3(material=m) for _ in range(mesh.nelem)]

    model = Model(mesh=mesh, elements=elements)

    # Generate Dirichlet BCs: fixed nodes on top
    ns = Nodeset(name="top", nodes=np.where(np.abs(mesh.coords[:, 1] - 1.0) < 1e-6)[0])
    mesh.nodesets[ns.name] = ns
    model.dirichlet_bcs.append(DirichletBC(nodeset=ns.name, dof=X, value=0.0, name="Top X"))
    model.dirichlet_bcs.append(DirichletBC(nodeset=ns.name, dof=Y, value=0.0, name="Top Y"))

    # Generate Neumann BCs: traction on bottom surface
    sides: list[tuple[int, int]] = []
    for e, nodes in enumerate(t):
        jlo = [node for node in nodes if isclose(mesh.coords[node, 1], -1.0)]
        if len(jlo) == 2:
            # 2 nodes at Y=1 indicate this is an edge
            edge = find_edge(elements[e], nodes, jlo)
            sides.append((e, edge))
    ss = Sideset("Top", sides=np.array(sides))
    mesh.sidesets[ss.name] = ss
    nbc = NeumannBC(sideset=ss.name, type=TRACTION, value=np.array((400.0e3, -300.0e3)))
    model.neumann_bcs.append(nbc)

    g = 9.81
    model.source = lambda x, y: np.array([0.0, -m.density * g])
    u, r = model.solve()
    scale = 0.25 / np.max(np.abs(u))
    U = np.linalg.norm(u, axis=1)
    print(np.amax(U))
    tplot(mesh.coords + scale * u, mesh.connect, U)

    from .done import exercise as exxx

    exxx(p, t)


def plate_with_hole(esize: float) -> tuple[NDArray, NDArray]:
    fd = lambda p: ddiff(drectangle(p, -1, 1, -1, 1), dcircle(p, 0, 0, 0.5))
    fh = lambda p: 0.05 + 0.3 * dcircle(p, 0, 0, 0.5)
    bbox = ((-1.0, 1.0), (-1.0, 1.0))
    fixed = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    p, t = distmesh2d(fd, fh, esize, bbox, fixed)
    return p, t


def uniform_plate(esize: float) -> tuple[NDArray, NDArray]:
    fd = lambda p: drectangle(p, -1, 1, -1, 1)
    fh = lambda p: huniform(p)
    bbox = ((-1.0, 1.0), (-1.0, 1.0))
    fixed = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    p, t = distmesh2d(fd, fh, esize, bbox, fixed)
    return p, t


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-s", type=float, default=0.05, help="Element size [default: %(default)s]")
    args = p.parse_args()
    exercise(esize=args.s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
