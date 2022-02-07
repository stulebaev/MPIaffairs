/* correlation.c */
#include <stdio.h>
#include "mpi.h"

#define MAX_ITEMS 100
#define ERROR_CODE_TAG 300

int main(int argc, char* argv[]) {
  int my_rank, numprocs;
  int error_flag = 0;
  int i, n_div, num_items;
  int *counts = NULL;
  int *displacements = NULL;
  float *x_array = NULL;
  float *y_array = NULL;
  float *x_array_local = NULL;
  float *y_array_local = NULL;
  float Mx, My, Sx, Sy, cov, rho;
  float local1, local2, local3, tempx, tempy;
  FILE *fp;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  if (argc < 2) {
    if (my_rank == 0) printf("Usage: correlation datafile\n");
    error_flag = 1;
    goto exit_label;
  }
  if (my_rank == 0) {
    if (!(fp = fopen(argv[1], "r"))) {
      printf("Can't open file %s\n", argv[1]);
      error_flag = 1;
    }
  }
  MPI_Bcast(&error_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (error_flag) goto exit_label;
  if (my_rank == 0) {
    x_array = (float*)malloc(MAX_ITEMS*sizeof(int));
    y_array = (float*)malloc(MAX_ITEMS*sizeof(int));
    i = 0;
    while (!feof(fp) && (i < MAX_ITEMS)) {
      fscanf(fp, "%f %f", &x_array[i], &y_array[i]);
      i++;
    }
    num_items = i;
  }
  MPI_Bcast(&num_items, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  counts = (int*)malloc(numprocs*sizeof(int));
  displacements = (int*)malloc(numprocs*sizeof(int));
  n_div = num_items/numprocs;
  for (i = 0; i < numprocs; i++) {
    counts[i] = n_div;
    displacements[i] = i*n_div;
  }
  counts[numprocs-1] += num_items%numprocs;
  x_array_local = (float*)malloc((counts[my_rank])*sizeof(int));
  y_array_local = (float*)malloc((counts[my_rank])*sizeof(int));
  MPI_Scatterv(x_array, counts, displacements, MPI_FLOAT, x_array_local,
               counts[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(y_array, counts, displacements, MPI_FLOAT, y_array_local,
               counts[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  local1 = x_array_local[0]; local2 = y_array_local[0];
  for (i = 1; i < counts[my_rank]; i++) {
    local1 += x_array_local[i];
    local2 += y_array_local[i];
  }
  MPI_Allreduce(&local1, &Mx, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local2, &My, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  Mx /= num_items; /* mean value of X */
  My /= num_items; /* mean value of Y */
  local1 = 0.0; local2 = 0.0; local3 = 0.0;
  for (i = 0; i < counts[my_rank]; i++) {
    tempx = x_array_local[i] - Mx;
    local1 += tempx*tempx;
    tempy = y_array_local[i] - My;
    local2 += tempy*tempy;
    local3 += tempx*tempy;
  }
  MPI_Reduce(&local1, &Sx, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local2, &Sy, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local3, &cov, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (my_rank == 0) {
    Sx = sqrt(Sx/num_items); /* deviation of X */
    Sy = sqrt(Sy/num_items); /* deviation of Y */
    cov /= num_items; /* covariance(X,Y) */
    rho = cov/Sx/Sy;
    printf("Correlation coefficient = %3.2lf\n", rho);
   }
exit_label:
  free(counts);
  free(displacements);
  free(x_array_local);
  free(x_array_local);
  free(x_array);
  free(y_array);
  MPI_Finalize();
  return(error_flag);
}
