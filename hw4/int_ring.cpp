#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


float ring(int token, int N, int rank, int w_size){
  double tt = MPI_Wtime();
  for(int i =0;i < N; i++){
  if (rank != 0) {
    MPI_Recv(&token, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    token += rank;
  }
  MPI_Send(&token, 1, MPI_INT, (rank + 1) % w_size, 0,
           MPI_COMM_WORLD);
  if (rank == 0) {
     MPI_Recv(&token, 1, MPI_INT, w_size - 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}
tt = MPI_Wtime() - tt;
return tt;
}
int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int w_size;
  MPI_Comm_size(MPI_COMM_WORLD, &w_size);
  int token = 0;
  int N = 1000;
  float tt = ring(token,N,rank,w_size);
  if(!rank){
  printf("Time: %e ms\n",tt * 1e3);
  printf("Latency: %e ms\n",tt/N * 1e3);
  printf("Bandwidth: %e GB/s\n", (1e6*N)/tt/1e9);}
  MPI_Finalize();
}
