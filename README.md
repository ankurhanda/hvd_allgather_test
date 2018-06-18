**How to run**

Assuming you have 16 GPUs on your machine you can run the test code as 

`mpirun -np 16 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH=/usr/local/nvidia/lib64/ python hvd_allgather.py`

 
