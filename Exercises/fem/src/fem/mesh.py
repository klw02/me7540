import logging
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Sequence
from typing import Type

import numpy as np
from numpy.typing import NDArray

from . import collections
from .block import TopoBlock
from .cell import Cell
from .collections import Map
from .typing import RegionSelector

logger = logging.getLogger(__name__)


@dataclass
class Mesh:
    coords: NDArray
    connect: NDArray
    blocks: list[TopoBlock]
    node_map: Map
    elem_map: Map
    block_elem_map: dict[int, int]
    nodesets: dict[str, list[int]] = field(default_factory=dict)
    sidesets: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    elemsets: dict[str, list[int]] = field(default_factory=dict)


class MeshBuilder:
    def __init__(self, nodes: Sequence[Sequence[int | float]], elements: list[list[int]]) -> None:
        self.assembled = False
        connected: set[int] = set([n for row in elements for n in row[1:]])
        allnodes: set[int] = set([int(row[0]) for row in nodes])
        if disconnected := allnodes.difference(connected):
            for n in disconnected:
                logger.error(f"Node {n} is not connected to any element")
            raise RuntimeError("Disconnected nodes detected")

        self.node_map: collections.Map = collections.Map([int(node[0]) for node in nodes])
        self.elem_map: collections.Map = collections.Map([int(elem[0]) for elem in elements])

        num_node: int = len(nodes)
        max_dim: int = max(len(n[1:]) for n in nodes)
        self.nodes: list[collections.Node] = []
        self.coords: NDArray = np.zeros((num_node, max_dim), dtype=float)
        for i, node in enumerate(nodes):
            xc = [float(x) for x in node[1:]]
            self.coords[i, : len(xc)] = xc
            ni = collections.Node(lid=i, gid=int(node[0]), x=xc)
            self.nodes.append(ni)

        num_elem: int = len(elements)
        max_elem: int = max(len(e[1:]) for e in elements)
        self.connect: NDArray = -np.ones((num_elem, max_elem), dtype=int)
        errors: int = 0
        for i, element in enumerate(elements):
            for j, gid in enumerate(element[1:]):
                if gid not in self.node_map:
                    errors += 1
                    logger.error(f"Node {j + 1} of element {i + 1} ({gid}) is not defined")
                    continue
                self.connect[i, j] = self.node_map.local(gid)
        if errors:
            raise ValueError("Stopping due to previous errors")

        # Meta data to store information needed for one pass mesh assembly
        self.metadata: dict[str, dict] = defaultdict(dict)

        self.edges: list[collections.Edge] = []
        self.blocks: list[TopoBlock] = []
        self.block_elem_map: dict[int, int] = {}
        self.elemsets: dict[str, list[int]] = defaultdict(list)
        self.nodesets: dict[str, list[int]] = defaultdict(list)
        self.sidesets: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def block(self, *, name: str, cell_type: Type[Cell], region: RegionSelector) -> None:
        blocks = self.metadata["blocks"]
        if name in blocks:
            raise ValueError(f"Topo block {name!r} already defined")
        blocks[name] = collections.BlockSpec(name=name, cell_type=cell_type, region=region)  # type: ignore

    def construct_sets(self) -> None:
        self.construct_nodesets()
        self.construct_elemsets()
        self.construct_sidesets()

    def build(self) -> Mesh:
        if self.assembled:
            raise ValueError("MeshBuilder is already assembled")
        self.assemble_blocks()
        self.detect_topology()
        self.assembled = True
        self.construct_sets()
        return Mesh(
            coords=self.coords,
            connect=self.connect,
            blocks=self.blocks,
            node_map=self.node_map,
            elem_map=self.elem_map,
            block_elem_map=self.block_elem_map,
            nodesets=self.nodesets,
            sidesets=self.sidesets,
            elemsets=self.elemsets,
        )

    def assemble_blocks(self) -> None:
        self.blocks.clear()
        self.block_elem_map.clear()
        assigned: set[int] = set()
        for name, spec in self.metadata.get("blocks", {}).items():
            # eids is the global elem index
            eids: list[int] = []
            for e, conn in enumerate(self.connect):
                p = self.coords[conn]
                x = p.mean(axis=0)
                if spec.region(x, on_boundary=False):
                    eids.append(e)

            # By this point, eids holds the local element indices of each element in the block
            mask = np.isin(eids, list(assigned))
            if np.any(mask):
                duplicates = ", ".join(str(eids[i]) for i, m in enumerate(mask) if m)
                raise ValueError(
                    f"Block {name}: attempting to assign elements {duplicates} "
                    "which are already assigned to other topo blocks"
                )
            assigned.update(eids)

            b = len(self.blocks)
            self.block_elem_map.update({eid: b for eid in eids})

            nids: set[int] = set()
            elements: list[list[int]] = []
            for eid in eids:
                nids.update(self.connect[eid])
                elem = [self.elem_map[eid]] + [self.node_map[n] for n in self.connect[eid]]
                elements.append(elem)

            nodes: list[list[int | float]] = []
            for nid in sorted(nids):
                node = [self.node_map[nid]] + self.coords[nid].tolist()
                nodes.append(node)

            block = TopoBlock(name, nodes, elements, spec.cell_type)
            self.blocks.append(block)

        # Check if all elements are assigned to a topo block
        num_elements = self.connect.shape[0]
        if unassigned := set(range(num_elements)).difference(assigned):
            s = ", ".join(str(_) for _ in unassigned)
            raise ValueError(f"Elements {s} not assigned to any element blocks")

    def detect_topology(self) -> None:
        """Detect boundary faces/edges for all blocks and elements."""

        # mapping from face (tuple of sorted node indices) -> list of (block no, local element no, local face no)
        edges: dict[tuple[int, ...], list[tuple[int, int, int]]] = defaultdict(list)

        # Step 1: iterate all blocks and all elements in each block
        for b, block in enumerate(self.blocks):
            for e, conn in enumerate(block.connect):
                for edge_no in range(block.cell_type.nedge):
                    ix = block.cell_type.edge_nodes(edge_no)
                    gids = tuple(sorted([block.node_map[_] for _ in conn[ix]]))
                    edges[gids].append((b, e, edge_no))

        # Step 2: identify faces that are only in one element → boundary
        self.edges.clear()
        edge_normals: dict[int, list[NDArray]] = defaultdict(list)
        for specs in edges.values():
            if len(specs) == 1:
                b, e, edge_no = specs[0]
                block = self.blocks[b]
                conn = block.connect[e]
                p = block.coords[conn]
                normal = block.cell_type.edge_normal(edge_no, p)
                gid = block.elem_map[e]
                lid = self.elem_map.local(gid)
                xd = block.cell_type.edge_centroid(edge_no, p)
                info = collections.Edge(
                    element=lid, x=xd.tolist(), edge=edge_no, normal=normal.tolist()
                )
                self.edges.append(info)
                for ln in block.cell_type.edge_nodes(edge_no):
                    gid = block.node_map[conn[ln]]
                    lid = self.node_map.local(gid)
                    edge_normals[lid].append(normal)

        for lid, normals in edge_normals.items():
            avg_normal = np.mean(normals, axis=0)
            node = self.nodes[lid]
            assert node.lid == lid
            assert node.gid == self.node_map[lid]
            node.normal = avg_normal.tolist()
            node.on_boundary = True
        return

    def nodeset(
        self, name: str, region: RegionSelector | None = None, nodes: list[int] | None = None
    ) -> None:
        if region is None and nodes is None:
            raise ValueError("Expected region or nodes")
        elif region is not None and nodes is not None:
            raise ValueError("Expected region or nodes, not both")
        nodesets = self.metadata["nodesets"]
        if name in [ns[0] for ns in nodesets.values()]:
            raise ValueError(f"Duplicate node set {name!r}")
        nodesets[f"nodeset-{len(nodesets)}"] = (name, region, nodes)

    def construct_nodesets(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before adding constructing node sets")
        self.nodesets.clear()
        name: str
        region: RegionSelector | None
        nodes: list[int] | None
        for name, region, nodes in self.metadata.get("nodesets", {}).values():
            if region is not None:
                for node in self.nodes:
                    if region(node.x, on_boundary=node.on_boundary):  # type: ignore
                        self.nodesets[name].append(node.lid)
                if name not in self.nodesets:
                    raise ValueError(f"{name}: could not find nodes in region")
            elif nodes is not None:
                for gid in nodes:
                    self.nodesets[name].append(self.node_map.local(gid))

    def elemset(self, name: str, region: RegionSelector) -> None:
        elemsets = self.metadata["elemsets"]
        if name in elemsets:
            raise ValueError(f"Duplicate element set {name!r}")
        elemsets[name] = region

    def construct_elemsets(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before adding constructing element sets")
        self.elemsets.clear()
        name: str
        region: RegionSelector
        for name, region in self.metadata.get("elemsets", {}).items():
            for e, conn in enumerate(self.connect):
                p = self.coords[conn]
                x = p.mean(axis=0)
                if region(x, on_boundary=False):  # type: ignore
                    self.elemsets[name].append(e)

    def sideset(self, name: str, region: RegionSelector) -> None:
        sidesets = self.metadata["sidesets"]
        if name in sidesets:
            raise ValueError(f"Duplicate side set {name!r}")
        sidesets[name] = region

    def construct_sidesets(self) -> None:
        if not self.assembled:
            raise ValueError("Assemble builder before adding constructing element sets")
        self.sidesets.clear()
        name: str
        region: RegionSelector
        for name, region in self.metadata.get("sidesets", {}).items():
            for edge in self.edges:
                if region(edge.x, on_boundary=True):  # type: ignore
                    self.sidesets[name].append((edge.element, edge.edge))
