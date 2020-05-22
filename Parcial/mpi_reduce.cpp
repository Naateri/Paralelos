/*
MPI_Reduce implementation
Same as mpi_reduce.c but examen required .cpp file
Author: Renato Postigo
compile:
mpicc -g -Wall mpi_reduce.c -o mpi_reduce
execute:
mpiexec -n <processes> mpi_reduce

*/
#include <stdio.h>
#include <mpi.h>

char operations[] = {'+', 'M', 'm'};
// operations: 
// + = sum
// M = max
// m = min

int max(int a, int b){
	if (a > b) return a;
	else return b;
}

int min(int a, int b){
	if (a < b) return a;
	else return b;
}

void MY_MPI_Reduce(int* input_data, int* output_data, char op, int dest_process,
				   MPI_Comm comm, int my_rank, int comm_sz){
	int output = 0;
	if (op == 'm') output = 999999;
	if (comm_sz <= 1){
		printf("Not enough processes\n");
		return;
	}
	
	if (my_rank != dest_process){
		// Send to process dest_process
		MPI_Send(input_data, 1, MPI_INT, dest_process, 0, comm);
	} else {
		// dest_process
		for(int src = 0; src < comm_sz; src++){
			// do not recieve if i am dest_process
			if (src == my_rank) continue;
			// Recieve message from dest_process
			MPI_Recv(input_data, 1, MPI_INT, src, 0, comm, MPI_STATUS_IGNORE);
			if (op == '+')
				output += *input_data;
			else if (op == 'M')
				output = max(output, *input_data);
			else if (op == 'm')
				output = min(output, *input_data);
		}
		
		*output_data = output;
	}
	
}

int main(){
	int comm_sz, my_rank;
	
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	int my_value = my_rank, total_value;
	
	MY_MPI_Reduce(&my_value, &total_value, '+', 0, MPI_COMM_WORLD, my_rank, comm_sz);
	//printf("my_rank %d\n", my_rank);
	
	if (my_rank == 0){
		printf("Total sum of values in %d processes is: %d\n", comm_sz, total_value);
	}
	
	MY_MPI_Reduce(&my_value, &total_value, 'M', 0, MPI_COMM_WORLD, my_rank, comm_sz);
	
	if (my_rank == 0){
		printf("Maximum of values in %d processes is: %d\n", comm_sz, total_value);
	}
	
	MY_MPI_Reduce(&my_value, &total_value, 'm', 0, MPI_COMM_WORLD, my_rank, comm_sz);
	
	if (my_rank == 0){
		printf("Minimum of values in %d processes is: %d\n", comm_sz, total_value);
	}
	
	MPI_Finalize();
	return 0;
}
