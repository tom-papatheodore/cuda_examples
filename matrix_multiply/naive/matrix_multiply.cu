#include <stdio.h>

const int N       = 4096; // Matrix size
const float A_val = 1.0f; // Values of all elements of A
const float B_val = 2.0f; // Values of all elements of B

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                             \
do{                                                                                      \
  cudaError_t cuErr = call;                                                              \
  if(cudaSuccess != cuErr){                                                              \
    printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr)); \
    exit(0);                                                                             \
  }                                                                                      \
}while(0)

/* ----------------------------------------------------------------------------
Matrix multiply kernel
    - Each CUDA thread computes 1 element of the matrix via dot product
          i.e., C[row,col] = SUM(from i=0,N){A[row,i]*B[i,col]}

    - 1D indexing (row-major) through matrix gives
          index = row * n + col

    - Based on 1D indexing, we have
          single_row_index = row * n + i
          single_col_index = i * n + col
---------------------------------------------------------------------------- */
__global__ void mat_mul(const float *A, const float *B, float *C, int n) {

  /* -----------------------------------------------
  These span all cols and rows of the matrices based
  on the values of the configuration parameters
  ----------------------------------------------- */
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  if((col < n) && (row < n)){

    float element = 0;

    for(int i=0; i<n; i++){
      element += A[row * n + i] * B[i * n + col]; // Dot product of row,column
	}

    C[row * n + col] = element;
  }
}

int main(int argc, char *argv[]){

  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;

  // Allocate memory for arrays h_A, h_B, h_C on host
  h_A = new float[N*N];
  h_B = new float[N*N];
  h_C = new float[N*N];

  // Initialize host arrays
  for (int i = 0; i < N*N; i++){
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0;
  }

  // Allocate memory for arrays d_A, d_B, d_C on device
  cudaErrorCheck( cudaMalloc(&d_A, N*N*sizeof(float)) );
  cudaErrorCheck( cudaMalloc(&d_B, N*N*sizeof(float)) );
  cudaErrorCheck( cudaMalloc(&d_C, N*N*sizeof(float)) );

  // Copy values from host arrays into device arrays
  cudaErrorCheck( cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice) );
  cudaErrorCheck( cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice) );

  /* -------------------------------------------------------------
      Set execution configuration parameters
          threads_per_block: number of CUDA threads per grid block
          blocks_in_grid   : number of blocks in grid
          (These are structs with 3 member variables x, y, z)
   ------------------------------------------------------------ */
  dim3 threads_per_block(16,16,1);
  dim3 blocks_in_grid(ceil(float(N)/threads_per_block.x), ceil(float(N)/threads_per_block.y), 1);

  // Launch kernel
  mat_mul<<<blocks_in_grid, threads_per_block>>>(d_A, d_B, d_C, N);

  // Check for errors in the kernel launch (e.g., invalid configuration parameters)
  cudaError_t cuErrSync = cudaGetLastError();

  // Check for errors on the device after control is returned to host
  cudaError_t cuErrAsync = cudaDeviceSynchronize();

  if(cuErrSync != cudaSuccess){
    printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrSync)); exit(0);
  }

  if(cuErrAsync != cudaSuccess){
    printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrAsync)); exit(0);
  }

  // Copy results back to host
  cudaErrorCheck( cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost) );

  // Verify results
  for(int i = 0; i < N*N; i++){ 
    if(h_C[i] != A_val*B_val*N){
      printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val*B_val*N);
      return -1;
    }
  }

  cudaErrorCheck( cudaFree(d_A) );
  cudaErrorCheck( cudaFree(d_B) );
  cudaErrorCheck( cudaFree(d_C) );

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  printf("Success!\n"); 

  return 0;
}
  
