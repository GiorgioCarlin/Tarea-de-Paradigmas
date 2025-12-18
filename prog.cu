#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>


// Llena una matriz n×n con valores aleatorios
void llenar_matriz(float *M, int n) {
    for (int i = 0; i < n * n; i++) {
        M[i] = rand() % 10;
    }
}

// Imprime una matriz n×n
void imprimir_matriz(const char *nombre, float *M, int n) {
    printf("\nMatriz %s:\n", nombre);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.2f ", M[i*n + j]);
        }
        printf("\n");
    }
}

/* ================= CPU ================= */

// Multiplicación de matrices CPU multicore
void producto_cpu(float *A, float *B, float *C, int n, int nt) {

    #pragma omp parallel for num_threads(nt)
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

/* ================= GPU ================= */

// Kernel GPU básico (memoria global)
__global__ void producto_gpu(float *A, float *B, float *C, int n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float suma = 0.0f;
        for (int k = 0; k < n; k++) {
            suma += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = suma;
    }
}

/* ================= MAIN ================= */

int main(int argc, char *argv[]) {

    if (argc != 4) {
        printf("Uso: ./prog <n> <nt> <ALG>\n");
        return 1;
    }

    int n   = atoi(argv[1]);  // tamaño matriz
    int nt  = atoi(argv[2]);  // threads CPU
    int ALG = atoi(argv[3]);  // algoritmo

    printf("n   = %d\n", n);
    printf("nt  = %d\n", nt);
    printf("ALG = %d\n", ALG);

    srand(time(NULL));

    size_t size = n * n * sizeof(float);

    float *A = (float*) malloc(size);
    float *B = (float*) malloc(size);
    float *C = (float*) malloc(size);

    if (!A || !B || !C) {
        printf("Error al reservar memoria\n");
        return 1;
    }

    llenar_matriz(A, n);
    llenar_matriz(B, n);

    /* ================= CPU ================= */

    if (ALG == 1) {

        printf("\nEjecutando producto matricial CPU multicore...\n");

        double t_inicio = omp_get_wtime();
        producto_cpu(A, B, C, n, nt);
        double t_fin = omp_get_wtime();

        double tiempo = t_fin - t_inicio;
        printf("Tiempo CPU: %f segundos\n", tiempo);

        FILE *f = fopen("resultados_cpu.txt", "a");
        if (f) {
            fprintf(f, "%d %f\n", n, tiempo);
            fclose(f);
        }
    }

    /* ================= GPU ================= */

    else if (ALG == 2) {

        printf("\nEjecutando producto matricial GPU (basico)...\n");

        float *d_A, *d_B, *d_C;

        cudaMalloc((void**)&d_A, size);
        cudaMalloc((void**)&d_B, size);
        cudaMalloc((void**)&d_C, size);

        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x,
                  (n + block.y - 1) / block.y);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        producto_gpu<<<grid, block>>>(d_A, d_B, d_C, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error CUDA: %s\n", cudaGetErrorString(err));
        }

        float tiempo_ms;
        cudaEventElapsedTime(&tiempo_ms, start, stop);
        printf("Tiempo GPU: %f ms\n", tiempo_ms);

        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

        FILE *f = fopen("resultados_gpu.txt", "a");
        if (f) {
            fprintf(f, "%d %f\n", n, tiempo_ms / 1000.0);
            fclose(f);
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    /* ================= GPU SM ================= */

    else if (ALG == 3) {
        printf("\nGPU con memoria compartida aun no implementado.\n");
    }

    else {
        printf("\nALG incorrecto, debe ser 1, 2 o 3.\n");
    }

    /* ================= DEBUG ================= */

    if (n <= 8) {
        imprimir_matriz("A", A, n);
        imprimir_matriz("B", B, n);
        imprimir_matriz("C", C, n);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
