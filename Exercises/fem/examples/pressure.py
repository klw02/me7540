import argparse
import sys
from typing import Sequence

import numpy as np

import fem


def exercise(esize: float = 0.05):
    class Everywhere(fem.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return True

    class Inside(fem.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            if on_boundary and abs(x[0]) < 0.8 and abs(x[1]) < 0.8:
                return True
            return False

    nodes, elements = fem.meshing.plate_with_hole(esize=esize)
    mesh = fem.mesh.Mesh(nodes=nodes, elements=elements)
    mesh.block(name="Block-1", region=Everywhere(), cell_type=fem.cell.Tri3)
    mesh.nodeset("Top Left", region=lambda x, on_boundary: x[0] < -0.99 and x[1] > 0.99)
    mesh.nodeset("Top Right", region=lambda x, on_boundary: x[0] > 0.99 and x[1] > 0.99)
    mesh.sideset("Inside", region=Inside())
    mesh.elemset("All", region=Everywhere())

    material = fem.material.LinearElastic(density=2400.0, youngs_modulus=30.0e9, poissons_ratio=0.3)
    model = fem.model.Model(mesh, name="Pressure")
    model.assign_properties(block="Block-1", element=fem.element.CPS3(), material=material)

    simulation = fem.simulation.Simulation(model)
    step = simulation.static_step()
    step.boundary(nodes="Top Right", dofs=[1], value=0.0)
    step.boundary(nodes="Top Left", dofs=[0, 1], value=0.0)
    step.pressure(sideset="Inside", magnitude=500e3)

    simulation.run()
    solution = simulation.csteps[0].solution

    u = solution.dofs.reshape((model.nnode, -1))
    U = np.linalg.norm(u, axis=1)
    print(np.amax(U))

    scale = 0.25 / np.max(np.abs(u))
    fem.plotting.tplot(model.coords + scale * u, model.connect, U)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-s", type=float, default=0.05, help="Element size [default: %(default)s]")
    args = p.parse_args()
    exercise(esize=args.s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
