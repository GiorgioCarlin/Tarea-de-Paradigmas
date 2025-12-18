#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Tamaño fijo por ahora
#define N 4

// Función para llenar una matriz con números aleatorios
void llenar_matriz(float M[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M[i][j] = (float)(rand() % 10); // números entre 0 y 9
        }
    }
}

// Función para imprimir una matriz
void imprimir_matriz(const char* nombre, float M[N][N]) {
    printf("\nMatriz %s:\n", nombre);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", M[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {

    // Verificación básica de argumentos
    if (argc != 4) {
        printf("Uso correcto: ./prog <n> <nt> <ALG>\n");
        return 1;
    }

    int n   = atoi(argv[1]);   // tamaño (no usado aún)
    int nt  = atoi(argv[2]);   // threads CPU (no usado aún)
    int ALG = atoi(argv[3]);   // algoritmo (no usado aún)

    printf("Parámetros recibidos:\n");
    printf("n   = %d\n", n);
    printf("nt  = %d\n", nt);
    printf("ALG = %d\n", ALG);

    // Inicializar semilla aleatoria
    srand(time(NULL));

    // Declarar matrices 4x4
    float A[N][N];
    float B[N][N];

    // Llenar matrices con valores aleatorios
    llenar_matriz(A);
    llenar_matriz(B);

    // Imprimir matrices
    imprimir_matriz("A", A);
    imprimir_matriz("B", B);

    return 0;
}
