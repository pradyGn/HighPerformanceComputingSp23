#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n, long* T, int nthr) {
  //int p = omp_get_num_threads();
  //int t = omp_get_thread_num();
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
  T[0] =0;
  prefix_sum[0] = 0;
  if (n == 0 ) return;
  #pragma omp parallel num_threads(nthr)
  {
      long sum =0;
      int t = omp_get_thread_num();
      #pragma omp for schedule(static)
      for (long i=0; i<n; i++){
          sum += A[i];
          T[t+1] = sum;
          prefix_sum[i+1] = sum;
      }
      T[t+1] = sum;
  }
  for (long i=0; i<nthr; i++){
  T[i+1] += T[i];
  }


  #pragma omp parallel num_threads(nthr)
  {
    #pragma omp for schedule(static) 
      for (long i=1; i<n; i++){
        long tid = omp_get_thread_num();
        prefix_sum[i] += T[tid];
      }
  }

}

int main() {

  long nthr[4] = {10, 100, 1000};
  for (long i=0; i<4; i++)
  {
  printf("\n");
  std::cout << "Number of threads = " << nthr[i] << std::endl;
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  long* T = (long*) malloc(nthr[i] * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N, T, nthr[i]);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);
  free(A);
  free(B0);
  free(B1);
  free(T);
  }
 return 0;
}