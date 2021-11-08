CC       = mpicc
CXX      = mpicxx

CXX_FLAGS = -fPIC -I../include  -std=c++11
#CXX_FLAGS = -fPIC -I../include -DVERBOSE_SIMPLE 
#CXX_FLAGS = -fPIC -I../include -DDEBUG -DLIME -NO_MPI_IO -g
LINK_FLAGS = -lmpi

SLIB = liblattice.so
OBJS= dslash.o lattice_fermion.o lattice_gauge.o invert.o  check.o load_gauge.o
TARGET= main

all: ${TARGET}

.SUFFIXES: .cpp .c
.cpp.o:
	${CXX} ${CXX_FLAGS} -o $@ -c $<

${SLIB}: ${OBJS} 
	${CXX} --shared $^ -o $@ 

${TARGET}: main.cpp ${SLIB}
	${CXX} ${CXX_FLAGS} -Wl,-rpath=./ ${LINK_FLAGS} -o $@  $< -L./ -llattice

fmt:
	@#cd .. && git ls-files -- '*.cpp' '*.h' | xargs clang-format -i -style=file
	find .. -name "*.cpp" -or -name "*.h" | xargs clang-format -i -style=file

run:
	mpirun -n 4 ./main  0.005  UNIT  24 24 24 96   12 12 24 48

clean:
	rm -rf ${SLIB} ${OBJS} ${TARGET}

