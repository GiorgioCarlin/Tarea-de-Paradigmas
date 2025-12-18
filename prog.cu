#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// Inclusión del runtime de CUDA
#include <cuda_runtime.h>

// Tamaño de tile para el kernel con memoria compartida
#define TILE 16

// Macro simple para comprobar errores de llamadas CUDA
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t e = (call);                                               \
        if (e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(e));                                  \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Rellenar matriz con flotantes aleatorios en [0,9]
void llenar_matriz(float *M, int n) {
    for (int i = 0; i < n * n; ++i) M[i] = (float)(rand() % 10);
}

// Imprimir matriz (muestra hasta 8x8 para evitar salida demasiado grande)
void imprimir_matriz(const char* nombre, float *M, int n) {
    int lim = (n < 8) ? n : 8;
    printf("\nMatriz %s (mostrando %dx%d):\n", nombre, lim, lim);
    for (int i = 0; i < lim; ++i) {
        for (int j = 0; j < lim; ++j) {
            printf("%8.2f ", M[i * n + j]);
        }
        printf("\n");
    }
}

// Multiplicación de matrices de referencia en CPU: C = A * B
void matmul_cpu(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Kernel GPU básico (memoria global) - se usa si ALG==2
__global__ void matmul_kernel_basic(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Kernel GPU con tiling y memoria compartida (GPUSm / ALG==3)
__global__ void matmul_kernel_tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int m = 0; m < (n + TILE - 1) / TILE; ++m) {
        // Cargar un tile de A y B en memoria compartida, con comprobación de límites
        int aRow = row;
        int aCol = m * TILE + tx;
        if (aRow < n && aCol < n) As[ty][tx] = A[aRow * n + aCol];
        else As[ty][tx] = 0.0f;

        int bRow = m * TILE + ty;
        int bCol = col;
        if (bRow < n && bCol < n) Bs[ty][tx] = B[bRow * n + bCol];
        else Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n) C[row * n + col] = sum;
}

// Comparar dos matrices; devuelve el número de discrepancias
int comparar_matrices(const float *X, const float *Y, int n) {
    int mismatches = 0;
    const float eps = 1e-3f;
    for (int i = 0; i < n * n; ++i) {
        float a = X[i], b = Y[i];
        if (fabsf(a - b) > eps) {
            mismatches++;
            if (mismatches <= 10) {
                int r = i / n, c = i % n;
                printf("Mismatch at (%d,%d): %f vs %f\n", r, c, a, b);
            }
        }
    }
    return mismatches;
}

// Obtener el tiempo actual en segundos
double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso correcto: ./prog <n> <nt> <ALG>\n");
        printf("ALG: 1=CPU, 2=GPU básico, 3=GPU shared-memory (GPUSm)\n");
        return 1;
    }

    int n   = atoi(argv[1]);   // tamaño
    int nt  = atoi(argv[2]);   // threads CPU (no usado aquí)
    int ALG = atoi(argv[3]);   // algoritmo

    printf("Parámetros recibidos: n=%d, nt=%d, ALG=%d\n", n, nt, ALG);

    srand(time(NULL));

    size_t bytes = sizeof(float) * n * n;
    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    float *C_ref = (float*)malloc(bytes);
    float *C_gpu = (float*)malloc(bytes);
    if (!A || !B || !C_ref || !C_gpu) {
        fprintf(stderr, "Error allocating host memory\n");
        return 1;
    }

    llenar_matriz(A, n);
    llenar_matriz(B, n);

    imprimir_matriz("A", A, n);
    imprimir_matriz("B", B, n);

    // Calcular la referencia en CPU (siempre se calcula para verificación y medida)
    double t0 = now_seconds();
    matmul_cpu(A, B, C_ref, n);
    double t1 = now_seconds();
    printf("CPU time: %.6f s\n", t1 - t0);

    if (ALG == 1) {
        // Ya está calculado (solo CPU)
        imprimir_matriz("C_CPU", C_ref, n);
        free(A); free(B); free(C_ref); free(C_gpu);
        return 0;
    }

    // Reservar memoria en el dispositivo (GPU)
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

    dim3 block, grid;

    if (ALG == 2) {
        // Kernel básico - escoger bloque de 16x16
        block = dim3(16, 16);
        grid = dim3((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

        cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        matmul_kernel_basic<<<grid, block>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        printf("GPU basic kernel time: %.3f ms\n", ms);
        CUDA_CHECK(cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    } else if (ALG == 3) {
        // Kernel con tiling y memoria compartida
        block = dim3(TILE, TILE);
        grid = dim3((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

        cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        matmul_kernel_tiled<<<grid, block>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        printf("GPU tiled (shared) kernel time: %.3f ms\n", ms);
        CUDA_CHECK(cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    } else {
        fprintf(stderr, "ALG debe ser 1, 2 o 3\n");
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
        free(A); free(B); free(C_ref); free(C_gpu);
        return 1;
    }

    // Verificar corrección
    int mism = comparar_matrices(C_ref, C_gpu, n);
    if (mism == 0) printf("Resultado correcto (0 mismatches)\n");
    else printf("Resultado INCORRECTO: %d mismatches\n", mism);

    imprimir_matriz("C_GPU", C_gpu, n);

    // Liberar recursos
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    free(A); free(B); free(C_ref); free(C_gpu);

    return 0;
}
