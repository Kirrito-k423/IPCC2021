CC       = mpiicc
CXX      = mpiicpc

CXX_FLAGS = -fPIC -I../include  -std=c++11 -xcore-avx2
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
	mpirun -n 16 ./main 0.005  ../data/ipcc_gauge_24_72  24 24 24 72  12 12 12 36

prof:
	mpirun -n 8 vtune -collect hotspots -trace-mpi -result-dir profiling ./main 0.005 ../data/ipcc_gauge_24_72 24 24 24 72  12 12 24 36

clean:
	rm -rf ${SLIB} ${OBJS} ${TARGET}

