**How to run**

Assuming you have 16 GPUs on your machine you can run the test code as 

`mpirun -np 16 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH=/usr/local/nvidia/lib64/ python hvd_allgather.py`

This will gather tensors from all the GPUs via `hvd.allgather()` and all the gathered tensors then will be broadcast to all GPUs. Therefore, at the end of this every GPU will have the same gathered tensor.
 
