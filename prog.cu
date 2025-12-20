// matmul.cu - Multiplicación de matrices en CUDA
// Versiones: CPU multicore, GPU básica, GPU con memoria compartida
// Uso: ./prog <n> <nt> <ALG>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

// Tamaño de tile para memoria compartida
#define TILE_SIZE 16

// Macro para verificar errores CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ==================== FUNCIONES AUXILIARES ====================

void llenar_matriz(float *M, int n) {
    for (int i = 0; i < n * n; i++) {
        M[i] = (float)(rand() % 10);
    }
}

int verificar_resultados(float *C1, float *C2, int n, float tolerancia = 1e-3) {
    int errores = 0;
    for (int i = 0; i < n*n; i++) {
        if (fabs(C1[i] - C2[i]) > tolerancia) {
            errores++;
            if (errores <= 3) {
                int fila = i / n;
                int col = i % n;
                printf("  Error en [%d,%d]: %.2f vs %.2f\n", 
                       fila, col, C1[i], C2[i]);
            }
        }
    }
    return errores;
}

// ==================== CPU MULTICORE ====================

void matmul_cpu_multicore(float *A, float *B, float *C, int n, int nt) {
    #pragma omp parallel for num_threads(nt) collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float suma = 0.0f;
            for (int k = 0; k < n; k++) {
                suma += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = suma;
        }
    }
}

// ==================== GPU BÁSICA ====================

__global__ void matmul_gpu_basica(float *A, float *B, float *C, int n) {
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (fila < n && col < n) {
        float suma = 0.0f;
        for (int k = 0; k < n; k++) {
            suma += A[fila * n + k] * B[k * n + col];
        }
        C[fila * n + col] = suma;
    }
}

// ==================== GPU CON MEMORIA COMPARTIDA ====================

__global__ void matmul_gpu_sm(float *A, float *B, float *C, int n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int fila = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float suma = 0.0f;
    
    for (int m = 0; m < (n + TILE_SIZE - 1) / TILE_SIZE; m++) {
        int aFila = fila;
        int aCol = m * TILE_SIZE + tx;
        if (aFila < n && aCol < n)
            tileA[ty][tx] = A[aFila * n + aCol];
        else
            tileA[ty][tx] = 0.0f;
        
        int bFila = m * TILE_SIZE + ty;
        int bCol = col;
        if (bFila < n && bCol < n)
            tileB[ty][tx] = B[bFila * n + bCol];
        else
            tileB[ty][tx] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            suma += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (fila < n && col < n) {
        C[fila * n + col] = suma;
    }
}

// ==================== FUNCIÓN PRINCIPAL ====================

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: ./prog <n> <nt> <ALG>\n");
        printf("  n   : tamaño matriz\n");
        printf("  nt  : threads CPU (solo ALG=1)\n");
        printf("  ALG : 1=CPU, 2=GPU, 3=GPUsm\n");
        return 1;
    }
    
    int n = atoi(argv[1]);
    int nt = atoi(argv[2]);
    int ALG = atoi(argv[3]);
    
    if (ALG < 1 || ALG > 3) {
        printf("ERROR: ALG debe ser 1, 2 o 3\n");
        return 1;
    }
    
    // Configuración inicial
    srand(time(NULL));
    
    size_t size = n * n * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_ref = (float*)malloc(size);
    
    if (!h_A || !h_B || !h_C || !h_C_ref) {
        printf("Error: memoria CPU\n");
        return 1;
    }
    
    // Rellenar matrices
    llenar_matriz(h_A, n);
    llenar_matriz(h_B, n);
    
    // Calcular referencia (solo para verificación)
    double tiempo_ref = 0.0;
    if (n <= 1024) {
        double inicio_ref = omp_get_wtime();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float suma = 0.0f;
                for (int k = 0; k < n; k++) {
                    suma += h_A[i*n + k] * h_B[k*n + j];
                }
                h_C_ref[i*n + j] = suma;
            }
        }
        tiempo_ref = omp_get_wtime() - inicio_ref;
    }
    
    // Ejecutar algoritmo seleccionado
    double tiempo_ejecucion = 0.0;
    
    if (ALG == 1) {
        // CPU multicore
        double inicio = omp_get_wtime();
        matmul_cpu_multicore(h_A, h_B, h_C, n, nt);
        tiempo_ejecucion = omp_get_wtime() - inicio;
        
        printf("CPU: n=%d, threads=%d, tiempo=%.6f s\n", n, nt, tiempo_ejecucion);
        
    } else if (ALG == 2 || ALG == 3) {
        // Versiones GPU
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size));
        CUDA_CHECK(cudaMalloc(&d_B, size));
        CUDA_CHECK(cudaMalloc(&d_C, size));
        
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
        
        dim3 block, grid;
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        if (ALG == 2) {
            // GPU básica
            block = dim3(16, 16);
            grid = dim3((n + block.x - 1) / block.x,
                       (n + block.y - 1) / block.y);
            matmul_gpu_basica<<<grid, block>>>(d_A, d_B, d_C, n);
        } else {
            // GPU con memoria compartida
            block = dim3(TILE_SIZE, TILE_SIZE);
            grid = dim3((n + TILE_SIZE - 1) / TILE_SIZE,
                       (n + TILE_SIZE - 1) / TILE_SIZE);
            matmul_gpu_sm<<<grid, block>>>(d_A, d_B, d_C, n);
        }
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float tiempo_ms = 0.0;
        CUDA_CHECK(cudaEventElapsedTime(&tiempo_ms, start, stop));
        tiempo_ejecucion = tiempo_ms / 1000.0;
        
        // Verificar errores
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess) {
            printf("ERROR CUDA: %s\n", cudaGetErrorString(kernelErr));
        }
        
        // Copiar resultado
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
        
        // Liberar GPU
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        printf("GPU-%d: n=%d, tiempo=%.3f ms\n", ALG, n, tiempo_ms);
    }
    
    // Verificar resultados si es necesario
    if (n <= 1024 && ALG != 1) {
        int errores = verificar_resultados(h_C, h_C_ref, n);
        if (errores > 0) {
            printf("  Errores: %d\n", errores);
        }
    }
    
    // Liberar memoria CPU
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    
    return 0;
}
