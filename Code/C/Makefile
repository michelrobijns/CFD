target = main.exe
objects = main.o mesh.o matrices.o linearAlgebra.o

cc = gcc
cFlags = -lm -O0 -march=native -pg

all: $(target)

%.o: %.c
	$(cc) $(cFlags) -c $<

$(target): $(objects)
	$(cc) $(cFlags) -o $@ $^

clean:
	rm -f $(target) *.o
