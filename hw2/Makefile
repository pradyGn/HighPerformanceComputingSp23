CXX = g++
CXXFLAGS = -std=gnu++11 -O3 -march=native
VALFLAGS = -g
RM = rm -f
MKDIRS = mkdir -p

TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

MMult1:MMult1.cpp
	$(CXX) $(CXXFLAGS) $^ && ./a.out

val_test01_solved:val_test01_solved.cpp
	$(CXX) $(VALFLAGS) $^ && valgrind --tool=memcheck  --leak-check=full ./a.out

val_test02_solved:val_test02_solved.cpp
	$(CXX) $(VALFLAGS) $^ && valgrind --tool=memcheck  --leak-check=full ./a.out

fast-sin:fast-sin.cpp
	$(CXX) $(CXXFLAGS) $^ && ./a.out