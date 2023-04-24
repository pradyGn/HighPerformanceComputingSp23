#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>

// Creates an array of random numbers. Each number has a value from 0 - 1
float *create_rand_nums(int num_elements) {
  float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (float)RAND_MAX);
    printf("Random element at pos %d is %f\n", i, rand_nums[i]);
  }
  return rand_nums;
}

float * compute_local_scan(float *array, int num_elements) {
  int i;
  for (i = 1; i < num_elements; i++) {
    array[i] += array[i-1];
  }
  return array;
}

float * add_offset(float *array, int num_elements, float *offset, int rank){
  float offset_num = 0;
  for (int j = 0; j<rank; j++){
    offset_num += offset[j];
  }
  int i;
  for (i = 0; i < num_elements; i++) {
    array[i] += offset_num;
  }
  return array;

}

nt main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: avg num_elements_per_proc\n");
    exit(1);
  }

  int num_elements_per_proc = atoi(argv[1]);
  // Seed the random number generator to get different results each time
  srand(time(NULL));

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  float *rand_nums = NULL;
  if (world_rank == 0) {
    rand_nums = create_rand_nums(num_elements_per_proc * world_size);
  }

  float *sub_rand_nums = (float *)malloc(sizeof(float) * num_elements_per_proc);
  assert(sub_rand_nums != NULL);

  MPI_Scatter(rand_nums, num_elements_per_proc, MPI_FLOAT, sub_rand_nums,
              num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);


  float * local_scan = compute_local_scan(sub_rand_nums, num_elements_per_proc);

  // Gather all partial averages down to all the processes
  float *sub_offsets = (float *)malloc(sizeof(float) * world_size);
  assert(sub_offsets != NULL);
  MPI_Allgather(&local_scan[num_elements_per_proc - 1], 1, MPI_FLOAT, sub_offsets, 1, MPI_FLOAT, MPI_COMM_WORLD);

 // for (int i = 0; i < world_size; i++){
 //   printf("Offset elements are %f\n", sub_offsets[i]);
 // }

  float *local_scan_with_offset = add_offset(local_scan, num_elements_per_proc, sub_offsets, world_rank);

  for (int i = 0; i < num_elements_per_proc; i++){
    printf("Scan element at pos %d is %f\n", (world_rank*num_elements_per_proc) + i, local_scan_with_offset[i]);
  }

  // Clean up
  if (world_rank == 0) {
    free(rand_nums);
  }
  free(sub_offsets);
  free(sub_rand_nums);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
