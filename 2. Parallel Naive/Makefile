TARGET = boids_naive_fixed.exe
SRC = boids_naive_fixed.cu

all: $(TARGET)

$(TARGET): $(SRC)
	nvcc -O2 -std=c++17 $(SRC) -o $(TARGET)

clean:
	del /Q $(TARGET)