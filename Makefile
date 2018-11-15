#===General Variables===#
CC=gcc
CFLAGS=-Wall -Wextra -g3 -Ofast -Wno-uninitialized

all: makeAll

makeAll: makeNeural makeMain
	$(CC) $(CFLAGS) neural_network.o main.o -o neurons -ldl -lm -lblas -llapack

makeMain: main.c 
	$(CC) $(CFLAGS) -c main.c -o main.o 

makeNeural: neural_network.c neural_network.h
	$(CC) $(CFLAGS) -c neural_network.c -o neural_network.o

.PHONY: clean

clean:
	rm -f *~ *.o
