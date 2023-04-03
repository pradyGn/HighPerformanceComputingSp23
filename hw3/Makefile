CXX = g++
CXXFLAGS = -std=gnu++11 -O3 -march=native -fopenmp

TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

gs2D-omp:gs2D-omp.cpp
	$(CXX) $(CXXFLAGS) $^ && ./a.out

omp-scan:omp-scan.cpp
	$(CXX) $(CXXFLAGS) $^ && ./a.out

jacobi2D-omp:jacobi2D-omp.cpp
	$(CXX) $(CXXFLAGS) $^ && ./a.out
