target = main.exe
objects = main.o mesh.o matrices.o linearAlgebra.o fast.o

cc = gcc
cFlags = -Wall -Wextra -Werror -lm -Ofast -march=native
#ATLASFlags = -lptlapack -lptf77blas -lptcblas -latlas
ATLASFlags = -lcblas -latlas

all: $(target)

%.o: %.c
	$(cc) $(cFlags) $(ATLASFlags) -c $<

$(target): $(objects)
	$(cc) $(cFlags) $(ATLASFlags) -o $@ $^

clean:
	rm -f $(target) *.o *.dat
