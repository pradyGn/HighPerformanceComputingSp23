# Makefile for val_test01.cpp and val_test02.cpp

val_test01_solved:
	g++ --std=gnu++11 -O3 val_test01_solved.cpp && ./a.out

val_test02_solved:
	g++ --std=gnu++11 -O3 val_test02_solved.cpp && ./a.out

valgrind_check_val_test02_solved:
	g++ --std=gnu++11 -O3 val_test02_solved.cpp && valgrind --tool=memcheck ./a.out

valgrind_check_val_test01_solved:
        g++ --std=gnu++11 -O3 val_test01_solved.cpp && valgrind --tool=memcheck ./a.out

fast-sin:
	g++ --std=gnu++11 -O3 fast-sin.cpp && ./a.out

MMult1:
	g++ -std=c++11 -O3 -march=native MMult1.cpp && ./a.out
