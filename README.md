
<center>
<img src="res/logo.png" width="350" height="280"/>
</center>

# cunuSHT
general (adjoint) spin-n SHTs.

cunuSHT provides functions to calculate spherical harmonic transforms for any uniform and non-uniform grid.

It can run on both, CPU and GPU, and can do this for (custom) geometries.

Operators:

 - `synthesis_general()`: takes SHT coefficients and (possibly non-uniform) grid points at which they are to be evaluated and returns the SHT transformed position space data (a map).
 - `adjoint_synthesis_general()`: this is the adjoint operation, takes a map with (possibly non-uniform grid points) and returns the SHT coefficients for a uniform grid. 
 - `gclm2lenmap()`: similar to synthesis_general, but automatically performs dlm2pointing, iff no pointing is provided
 - `lenmap2gclm()`: similar to adjoint_synthesis_general, but again, perfroms dlm2pointing if needed.
 - `dlm2pointing()`: calculates the non-uniform grid points (pointing) from the coefficients of a deflection field (dlm).


## Application

Highly accurate remapping of pixels on the sphere, as shown in the following examples.
<center>
Simulated movement of Jupiters clouds
<img src="res/jupiter.gif" width="650" height="500"/>
Repeated deflection of the cosmic microwave background with random Gaussian Lambda-CDM deflection fields.
<img src="res/deffield.gif" width="550" height="400"/>
</center>


## Installation

Currently in 2 steps:

Enter the `cunusht/c` folder, and compile the C and CUDA library, and install the python module via the `pyproject.toml`

```
cd cunusht/c
pip install .
```

Then, go to the root directory, and install `cunusht`:

```
cd ./../../
python3 setup.py install
```

## usage

See our tutorials in [first steps](https://github.com/Sebastian-Belkner/cunuSHT/tree/main/first_steps).

The interface works as follows.

Set parameters,

```
lmax, mmax = 2047, 2047
geominfo = ('gl',{'lmax': lmax})
kwargs = {
    'geominfo_deflection': geominfo,
    'nuFFTtype': 2, # or 1 if you want to use adjoint_synthesis_general()
    'epsilon': 1e-7,
}
```

Construct your transformer,
```
import cunusht
t = cunusht.get_transformer(backend='GPU')(**kwargs)
```

 - alm are the SHT coefficients for which the map should be calculated
 - loc are the positions (theta, and phi) of the map for which the map should be calculated
 - pointmap is the output array
```
res = t.synthesis_general(lmax=lmax, mmax=mmax, alm=coef, loc=loc, pointmap=pointmap, verbose=True)
```

Or, if you want to calculate the adjoint (not how nuFFTtype changes here!)

```
lmax, mmax = 2047, 2047
geominfo = ('gl',{'lmax': lmax})
kwargs = {
    'geominfo_deflection': geominfo,
    'nuFFTtype': 1, # or 2 if you want to use synthesis_general()
    'epsilon': 1e-7,
}
import cunusht
t = cunusht.get_transformer(backend='GPU')(**kwargs)

res = t.adjoint_synthesis_general(lmax=lmax, mmax=mmax, pointmap=m, loc=loc, alm=alm)
```

## Lensing Convenience Functions
We provide convenience function for CMB weak lensing purposes.

### gclm2lenmap()
The function depends on,
- gclm: the SHT coefficients at the uniform grid points
- ptg: the (non-uniform) grid points (the pointing) for which the pointmap should be evaluated
- dlm(_scaled): the deflection field that is used to calculate pointing, iff ptg is not provided. Note, for the GPU backend, this is dlm_scaled and must be `dlm * np.sqrt(1/l(l+1))`.
- lmax: the pointmap maximum SHT multipole
- mmax: the dlm_scaled mmax
- lenmap: the output

Choose your parameters, then
```
res = t.gclm2lenmap(gclm=cp.array(coef), dlm_scaled=cp.zeros(shape=coef.shape), lmax=lmax, mmax=lmax, lenmap=pointmap)
```

### lenmap2gclm()
This is the adjoint (inverse) operation of `gclm2lenmap()`, if lenmap is not quadrature weighted (is multiplied by the maginification matrix, and quadrature weighted).

Similar to above,

```
t.lenmap2gclm(lenmap, dlm_scaled=dlm_scaled, lmax=lmax, mmax=lmax, epsilon=epsilon, nthreads=10 gclm=gclm)
```