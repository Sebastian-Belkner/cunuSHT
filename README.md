# pySHT
general spin-n SHTs on CPU and GPU backend



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