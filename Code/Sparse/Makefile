target = main.exe
objects = main.o mesh.o matrices.o linearAlgebra.o fast.o sparse.o

cc = gcc
cFlags = -Wall -Wextra -Werror -lm -O0
#cFlags = -Wall -Wextra -Werror -lm -fopenmp -Ofast -march=native
ATLASFlags = -lptlapack -lptf77blas -lptcblas -latlas

all: $(target)

%.o: %.c
	$(cc) $(cFlags) $(ATLASFlags) -c $<

$(target): $(objects)
	$(cc) $(cFlags) $(ATLASFlags) -o $@ $^

clean:
	rm -f $(target) *.o
