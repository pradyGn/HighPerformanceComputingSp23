#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#ifdef _OPENMP
    #include <omp.h>
#endif

using namespace std;

// Main Function

int main(int argc, char** argv)
{
    for (long N=1; N<130; N+=40){
        double h = 1.0/(N+1);
        long n = N*N;
        printf("---------------------------------------------------------\n");
        printf("   N     #Threads     Time elapsed(s)     Speedup\n");
        printf("---------------------------------------------------------\n");

        // Dynamically allocate memory using malloc()
        double* f = (double*) malloc(n*sizeof(double));
        double* u = (double*) malloc(n*sizeof(double));
        double* u_new = (double*) malloc(n*sizeof(double));

        for (long n_thread=1; n_thread<=31; n_thread+=3){
            double singtime = 1.0, t;
            #ifdef _OPENMP
                t = omp_get_wtime();
            #endif

            for (long i=0; i < n; i++){
                f[i] = 1.0;
                u[i] = 0.0;
                u_new[i] = 0.0;
            }

            for (long c=0; c<=500; c++){
                #pragma omp parallel for num_threads(n_thread) shared(u, u_new)
                for (long i=0; i< n; i++) u_new[i] = 0.25*((h*h*f[i])+ u[i+1] + u[i-1] + u[i+N] + u[i-N]);
                for (long i=0; i<n; i++)  u_new[i] = 0.25*((h*h*f[i])+ u_new[i+1] + u_new[i-1] + u_new[i+N] + u_new[i-N]);
                for (long i=0; i<n; i++)  u[i] = u_new[i];
            }

            #ifdef _OPENMP
                t = omp_get_wtime()-t;
            #endif

            if(n_thread == 1)
                singtime = t;
            printf(" %5d    %5d       %10f        %10f \n", N, n_thread, t, singtime/t);
        }

        free(f);
        free(u);
        free(u_new);
    }
    return 0;
}