#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>

int matrix_orders[] = {1024, 2048, 4096, 8192};//, 16384};
//4GB RAM can't handle the algorithm with 16384x16384 matrices (over 4GB in space)
double *matrix, *vector, *mat_result;

double get_random(double max){
	srand( (unsigned) time(0));
	return (max / RAND_MAX) * rand();
}

//As seen in Peter Pacheco's An Introduction to Parallel Programming
void matrix_vector_mult(double local_A[], double local_X[], double local_Y[],
						int local_m, int n, int local_n, MPI_Comm comm){
	double* x;
	int local_i, j;
	int local_ok = 1;
	
	x = (double*) malloc(n*sizeof(double));
	MPI_Allgather(local_X, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, comm);
	
	for(local_i = 0; local_i < local_m; local_i++){
		local_Y[local_i] = 0.0;
		for(j = 0; j < n; j++){
			local_Y[local_i] += local_A[local_i*n+j] * x[j];
		}
	}
	
	free(x);
}

//Store matrix as 1 dimensional array

double* create_matrix(double* matrix, int size, int rank, int fill_with_zeros){
	if (fill_with_zeros == 1){
		matrix = (double*) calloc((size*size),sizeof(double));
	} else {
		matrix = (double*) malloc((size*size)*sizeof(double));
		if (rank == 0){
			for (int i = 0; i < size; i++){
				for(int j = 0; j < size; j++){
					matrix[i*size + j] = get_random(550.0);
				}
			}
		}
	}
	
	return matrix;
}

void multiply_vector(int size, int comm_sz, int my_rank){
	double local_start, local_end, local_elapsed, elapsed;
	
	matrix = create_matrix(matrix, size, my_rank, 0);
	vector = (double*) malloc(size*sizeof(double));
	
	if (my_rank == 0){
		
		printf("Comm size: %d\n", comm_sz);
		printf("Size: %d\n", size);
		
		//Only process 0 creates matrix and vector
		//All processes need to separate memory
		
		for(int i = 0; i < size; i++){
			vector[i] = get_random(550.0);
		}
		
	}
	
	MPI_Bcast(matrix, size*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(vector, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	mat_result = create_matrix(mat_result, size, my_rank, 1);
	
	int local_n;
	double *local_A, *local_X, *local_Y;
	local_n = size/comm_sz;
	
	//printf("Process %d of %d\n", my_rank, comm_sz);
	
	local_A = (double*) (malloc(local_n * size * sizeof(double)));
	int local_i = 0;
	
	int first_row = my_rank*local_n, last_row = (my_rank+1)*local_n;
	
	for(int i = first_row; i < last_row; i++){
		for(int j = 0; j < size; j++){
			int local_j = 0;
			local_A[local_i * size + local_j] = matrix[i * size + j];
			local_j++;
		}
		local_i++;
	}
	
	local_X = (double*) (malloc(local_n * sizeof(double)));
	
	local_i = 0;
	for(int i = my_rank*local_n; i < (my_rank+1)*local_n; i++){
		local_X[local_i] = vector[i];
		local_i++;
	}
	
	local_Y = (double*) (malloc(local_n * size * sizeof(double)));
	
	//printf("Size: %d\n", size);
	//start time measure
	MPI_Barrier(MPI_COMM_WORLD);
	local_start = MPI_Wtime();
	
	matrix_vector_mult(local_A, local_X, local_Y, local_n, size, local_n, MPI_COMM_WORLD);
	
	//end time measure
	local_end = MPI_Wtime();
	local_elapsed = local_end - local_start;
	
	MPI_Allreduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	
	if (my_rank == 0){
		printf("Elapsed time = %f milliseconds\n", elapsed*1000);
	}
	
	free(matrix);
	free(mat_result);
	free(vector);
	free(local_A);
	free(local_X);
	free(local_Y);
	
}


int main(int argc, char *argv[]) {
	
	int my_rank, comm_sz;
	
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	
	//printf("Comm size: %d\n", comm_sz);
	
	/*if (comm_sz >= 8){
		for(int i = 0; i < 4; i++){
			//multiply_vector
			multiply_vector(matrix_orders[i], comm_sz, my_rank);
		}
	} else {*/
		for(int i = 0; i < 4; i++){
			//multiply_vector
			multiply_vector(matrix_orders[i], comm_sz, my_rank);
		}
	//}

	MPI_Finalize();
	
	return 0;
}

