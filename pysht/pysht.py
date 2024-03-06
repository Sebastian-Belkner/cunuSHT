from __future__ import annotations


from pysht.deflection.GPU_nufft_transformer import GPU_cufinufft_transformer
from pysht.deflection.CPU_nufft_transformer import CPU_finufft_transformer, CPU_DUCCnufft_transformer, CPU_Lenspyx_transformer

from pysht.sht.CPU_sht_transformer import CPU_SHT_DUCC_transformer, CPU_SHT_SHTns_transformer
from pysht.sht.GPU_sht_transformer import GPU_SHTns_transformer
from pysht.visitor import transform, transform3d
from pysht.geometry import Geom

class CPU_SHT_Transformer:
    def build(self, solver):
        if solver in ['shtns']:
            return CPU_SHT_SHTns_transformer
        elif solver in ['ducc']:
            return CPU_SHT_DUCC_transformer
        else:
            assert 0, "Solver not found: {}".format(solver)
        
class CPU_nuFFT_Transformer:
    def build(self, solver):
        if solver in ['duccnufft']:
            return CPU_DUCCnufft_transformer#(shttransformer_desc='ducc')
        elif solver in ['lenspyx']:
            return CPU_Lenspyx_transformer#(shttransformer_desc='ducc')
        elif solver in ['finufft']:
            return CPU_finufft_transformer#(shttransformer_desc='ducc')
        else:
            assert 0, "Solver not found: {}".format(solver)

class GPU_SHT_Transformer:
    def build(self, solver):
        if solver in ['shtns']:
            return GPU_SHTns_transformer
        else:
            assert 0, "Solver not found: {}".format(solver)
        
class GPU_nuFFT_Transformer:
    def build(self, solver):
        if solver in ['cufinufft']:
            return GPU_cufinufft_transformer#(shttransformer_desc='shtns')
        else:
            assert 0, "Solver not found: {}".format(solver)


def get_geom(geometry: tuple[str, dict]=('healpix', {'nside':2048})):
    r"""Returns sphere pixelization geometry instance from name and arguments

        Note:
            Custom geometries can be defined following lenspyx.remapping.utils_geom.Geom

    """
    geo = getattr(Geom, '_'.join(['get', geometry[0], 'geometry']), None)
    if geo is None:
        assert 0, 'Geometry %s not found, available geometries: '%geometry[0] + Geom.get_supported_geometries()
    return geo(**geometry[1])


def set_transformer(transf):
    assert 0, "implement if needed"


def get_transformer(solver, mode, backend):
    if backend in ['CPU']:
        if mode in ['SHT']:
            return transform(solver, CPU_SHT_Transformer())
        elif mode in ['nuFFT']:
            return transform(solver, CPU_nuFFT_Transformer())
        else:
            assert 0, "Mode not found"
    elif backend in ['GPU']:
        if mode in ['SHT']:
            return transform(solver, GPU_SHT_Transformer())
        elif mode in ['nuFFT']:
            return transform(solver, GPU_nuFFT_Transformer())
        else:
            assert 0, "Mode not found"
    else:
        assert 0, "Backend not found"


@transform.case(str, CPU_SHT_Transformer)
def _(solver, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(solver)

@transform.case(str, CPU_nuFFT_Transformer)
def _(solver, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(solver)

@transform.case(str, GPU_SHT_Transformer)
def _(solver, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(solver)

@transform.case(str, GPU_nuFFT_Transformer)
def _(solver, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(solver)