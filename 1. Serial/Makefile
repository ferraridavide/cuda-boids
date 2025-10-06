CC = gcc
CFLAGS = -lm
SRC = boids.c
OUT = boids.exe

all: $(OUT)

$(OUT): $(SRC)
	"$(CC)" $(SRC) $(CFLAGS) -o $(OUT)

clean:
	del /Q $(OUT)