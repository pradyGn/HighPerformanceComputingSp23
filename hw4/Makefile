INIT = module load openmpi/gcc/4.0.5
CXX = mpic++ # or mpicxx
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp
PROCESSES = 3
TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

int_ring:int_ring.cpp
	$(INIT) && $(CXX) $(CXXFLAGS) $< $(LIBS) -o $@ && mpirun -np $(PROCESSES) $@

int_ring_arr:int_ring_arr.cpp
	$(INIT) && $(CXX) $(CXXFLAGS) $< $(LIBS) -o $@ && mpirun -np $(PROCESSES) $@

mpi-scan:mpi-scan.cpp
	$(INIT) && $(CXX) $(CXXFLAGS) $< $(LIBS) -o $@ && mpirun -np $(PROCESSES) $@ 2
