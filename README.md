# INF0188 Tarea 2: Multiplicación de Matrices en CUDA

## Descripción
Implementación del producto de matrices cuadradas en CUDA con 3 versiones:
1. **CPU multicore**: Paralelización con OpenMP
2. **GPU básica**: Implementación directa usando memoria global
3. **GPU con memoria compartida**: Optimización mediante tiling (16×16)

**Nota**: Todas las versiones se ejecutan con `srun`

## Estructura
matmul_COMPA/
matmul.cu # Código fuente


Makefile # Compilación


README.md # Este archivo

## Compilación

make clean 


make

## Uso

### CPU multicore (8 threads)
srun -p A4000 --gpus=1 --container-name=cuda --container-workdir=${PWD} ./prog <n> 8 1

### GPU básica
srun -p A4000 --gpus=1 --container-name=cuda --container-workdir=${PWD} ./prog <n> 1 2

### GPU con memoria compartida
srun -p A4000 --gpus=1 --container-name=cuda --container-workdir=${PWD} ./prog <n> 1 3
Parámetros:

n: Tamaño matriz (n × n)

nt: Threads CPU (solo ALG=1)

ALG: 1=CPU, 2=GPU básica, 3=GPU con memoria compartida

## Hardware (Patagon)
### GPU - NVIDIA RTX A4000

Capacidad cómputo: 8.6

Multiprocesadores: 48

Memoria global: 16 GB

Memoria compartida/bloque: 48 KB

### CPU

OpenMP disponible: Sí

Máximo threads: 8

# Resultados

## Tiempos de Ejecución
n	CPU (8 threads)	GPU básica	GPU con memoria compartida
256	3.932 ms	0.132 ms	0.126 ms
512	46.690 ms	0.316 ms	0.261 ms
1024	422.905 ms	1.917 ms	1.486 ms

## Speedup vs CPU
n	GPU básica	GPU con memoria compartida
256	29.8×	31.2×
512	147.8×	178.9×
1024	220.6×	284.6×

# Análisis
## CPU Multicore
Paralelización OpenMP con 8 threads

Buen rendimiento para matrices pequeñas

Escalabilidad limitada por cores físicos

## GPU Básica
Kernel simple, acceso coalescente a memoria

Speedup significativo (29.8× a 220.6×)

Limitado por accesos repetidos a memoria global

## GPU con Memoria Compartida
Tiling 16×16 optimiza reuso de datos

4-17% más rápido que GPU básica

Reducción accesos a memoria global mediante caché

Conclusiones
El análisis de rendimiento demuestra que las GPUs  proporcionan una aceleración significativa para la multiplicación de matrices, alcanzando speedups de hasta 284× sobre la implementación CPU multicore. La optimización mediante memoria compartida mejora consistentemente el rendimiento en un 4-17% respecto a la versión GPU básica, evidenciando la importancia del reuso de datos mediante tiling. Se observa que el speedup aumenta con el tamaño del problema, ya que el overhead de transferencia se vuelve menos significativo en matrices grandes. Para problemas de tamaño pequeño (n < 256), la implementación CPU mantiene competitividad debido a su menor overhead de ejecución, mientras que para problemas de mayor escala, la paralelización masiva de las GPUs resulta claramente ventajosa.

