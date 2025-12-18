#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

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

int main(int argc, char *argv[]) {

    if (argc != 4) {
        printf("Uso: ./prog <n> <nt> <ALG>\n");
        return 1;
    }

    int n   = atoi(argv[1]);
    int nt  = atoi(argv[2]);
    int ALG = atoi(argv[3]);

    printf("n   = %d\n", n);
    printf("nt  = %d\n", nt);
    printf("ALG = %d\n", ALG);

    srand(time(NULL));

    // Reservar memoria
    float *A = (float*) malloc(n * n * sizeof(float));
    float *B = (float*) malloc(n * n * sizeof(float));
    float *C = (float*) malloc(n * n * sizeof(float));

    if (!A || !B || !C) {
        printf("Error al reservar memoria\n");
        return 1;
    }

    // LLENAR MATRICES
    llenar_matriz(A, n);
    llenar_matriz(B, n);

    // Ejecutar algoritmos

    //CPU
    if (ALG == 1) {
        printf("\nEjecutando producto matricial CPU multicore...\n");

        double t_inicio = omp_get_wtime();
        producto_cpu(A, B, C, n, nt);
        double t_fin = omp_get_wtime();

        double tiempo = t_fin - t_inicio;

        printf("Tiempo CPU: %f segundos\n", tiempo);

        // Guardar resultados en archivo
        FILE *f = fopen("resultados_cpu.txt", "a");
        if (f != NULL) {
            fprintf(f, "%d %d %f\n", n, nt, tiempo);
            fclose(f);
        } else {
            printf("No se pudo abrir el archivo de resultados\n");
        }
    }
    //GPU
    else if (ALG == 2) {
        printf("\nEl producto en GPU aun no esta implementado...\n");
       
    } 
    //GPUsm
    else if (ALG == 3) {
        printf("\nEl producto en GPUsm aun no esta implementado...\n");
        
    } else {
        printf("\nALG incorrecto, debe ser igual a 1, 2 o 3.\n");
    }

    // Imprimir solo si es pequeño
    if (n <= 8 && ALG == 1) {
        imprimir_matriz("A", A, n);
        imprimir_matriz("B", B, n);
        imprimir_matriz("C", C, n);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
