from __future__ import annotations


from pysht.deflection import CPU_finufft, GPU_cufinufft, CPU_duccnufft, CPU_ducc
from pysht.visitor import transform, transform3d
from pysht.geometry import Geom

class CPU_Transformer:
    def build(self, solver):
        if solver in ['duccnufft']:
            return CPU_duccnufft.base()
        elif solver in ['ducc']:
            return CPU_ducc.base()
        elif solver in ['finufft']:
            return CPU_finufft.base()


class GPU_Transformer:
    def build(self, solver):
        if solver in ['cufinufft']:
            return GPU_cufinufft.base()
        elif solver in ['shtns']:
            assert 0, "not implemented"


def get_geom(geometry: tuple[str, dict]=('healpix', {'nside':2048}), backend='CPU'):
    r"""Returns sphere pixelization geometry instance from name and arguments

        Note:
            Custom geometries can be defined following lenspyx.remapping.utils_geom.Geom

    """
    geo = getattr(Geom, '_'.join(['get', geometry[0], 'geometry']), None)
    if geo is None:
        assert 0, 'Geometry %s not found, available geometries: '%geometry[0] + Geom.get_supported_geometries()
    return geo(**geometry[1], backend=backend)


def set_transformer(transf):
    assert 0, "implement if needed"


def get_transformer(solver, backend):
    if backend in ['CPU']:
        return transform(solver, CPU_Transformer())
    elif backend in ['GPU']:
        return transform(solver, GPU_Transformer())



@transform.case(str, CPU_Transformer)
def _(solver, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(solver)

@transform.case(str, GPU_Transformer)
def _(solver, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(solver)