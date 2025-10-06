TARGET = boids_fast.exe
SRC = boids_fast.cu

all: $(TARGET)

$(TARGET): $(SRC)
	nvcc -O3 -lineinfo -std=c++17 $(SRC) -o $(TARGET)

clean:
	del /Q $(TARGET)