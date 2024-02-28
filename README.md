# pySHT
general spin-n SHTs on CPU and GPU backend


## Rusty @ Simons

```
salloc -p gpu --gpus=1 -C v100 -c 1
ssh sbelkner@<workergpuX>
remember workerid!
module load python gcc cuda
./start_jupyter (don't miss half the token..)
connect via VS Code kernel
```

## Ygdrassil
Quickhelp to activate GPU on jupyter notebook in VS code


Activate conda environment, and load modules before starting kernel,

```
conda activate shtgpu
source load_modules
```

Allocate GPU,
```
salloc --gpus 1 --partition=shared-gpu --time=320:00
```

Run script for starting server,

```
./start_jupyter.sh
```

In VS Code, choose exsiting server with URL.
Finally, in notebook, choose shtgpu kernel.


To install shtns, activate gpu node, load compatible modules (CUDAcore needs gcc <=10, i.e: module load GCC/9.3.0)

There appears to be two shtgpu conda environs. One is on login node, the other is on gpu node. Install stuff into the gpu node version, as only there shtns and cufinufft installations succeed.