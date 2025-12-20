TARGET = prog
SOURCE = matmul.cu

# Ruta absoluta
WORK_DIR = /home/info188/aleal/matmul_COMPA

# Comando de compilación
COMPILE_CMD = srun -p A4000 --gpus=1 --container-name=cuda --container-workdir>

# Flags de compilación
NVCC_FLAGS = -O3 -Xcompiler -fopenmp -std=c++11 -arch=sm_86

# Compilar
all: $(TARGET)

$(TARGET): $(SOURCE)
        $(COMPILE_CMD) $(NVCC_FLAGS) $(SOURCE) -o $(TARGET)

# Limpiar
clean:
        rm -f $(TARGET)

# Pruebas rápidas
test_cpu: $(TARGET)
        srun -p A4000 --gpus=1 --container-name=cuda $(WORK_DIR)/$(TARGET) 256>

test_gpu: $(TARGET)
        srun -p A4000 --gpus=1 --container-name=cuda $(WORK_DIR)/$(TARGET) 256>

test_gpu_sm: $(TARGET)
        srun -p A4000 --gpus=1 --container-name=cuda $(WORK_DIR)/$(TARGET) 256>

.PHONY: all clean test_cpu test_gpu test_gpu_sm

