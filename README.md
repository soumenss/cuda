#cuda

As I was using microway server to run the code, could not compile it offline. So here I am adding the instructions to run this code.

Create a folder and copy the TiledMatrixMul.cu file in that folder. Open terminal and copile the code using nvcc.



nvcc TiledMatrixMul.cu -o TiledMatrixMul




An execution file will be created and now we can use this file to run the code.



./TiledMatrixMul 5 3 6




This will print 

Matrix A:
0.84 0.39 0.78 
0.80 0.91 0.20 
0.34 0.77 0.28 
0.55 0.48 0.63 
0.36 0.51 0.95 
Matrix B:
0.92 0.64 0.72 0.14 0.61 0.02 
0.24 0.14 0.80 0.16 0.40 0.13 
0.11 1.00 0.22 0.51 0.84 0.61 
Matrix C:
0.95 1.37 1.09 0.58 1.33 0.54 
0.97 0.83 1.35 0.36 1.02 0.25 
0.52 0.60 0.92 0.31 0.74 0.28 
0.69 1.05 0.92 0.48 1.06 0.46 
0.56 1.25 0.88 0.62 1.23 0.66 




While computing throughput we need to compile the code including -G to include debug symbols in the binary.



nvcc -G TiledMatrixMul.cu -o TiledMatrixMul



And for performace monitoring 


nvprof ./TiledMatrixMul 5 3 6
