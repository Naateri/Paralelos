/*
Ping pong of elements
Only 2 ranks
rank 0: even numbers
rank 1: odd numbers

compile:
mpicc -g -Wall ping_pong_elements.c -o ping_pong
execute:
mpiexec -n 2 ./ping_pong

*/
#include <stdio.h>
#include <string.h>
#include <mpi.h>

void get_input(int my_rank, int comm_sz, int* limit){
	if (my_rank == 0){
		printf("Insert limit ");
		scanf("%d", limit);
		printf("limit is %d\n", *limit);
		MPI_Send(limit, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
	} else {
		MPI_Recv(limit, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//limit = local_limit;
		printf("Process %d, limit %d\n", my_rank, *limit);
		//local_num = 1;
	}
}

int main(){
	int comm_sz, my_rank, limit, local_limit, num = 0, local_num = 0;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	int send;
	
	get_input(my_rank, comm_sz, &limit);
	
	if (my_rank == 0) send = 1;
	else send = 0;
	
	printf("Process %d, send = %d\n", my_rank, send);

	int limit_even = limit % 2;
	
	printf("------------------------------\n");
	
	while (num < limit){
		//printf("num %d\n", num);
		if (my_rank == 0){
			
			if (send == 1){
				//num = local_num;
				num++;
				printf("Process %d of %d, value = %d\n", my_rank, comm_sz, num);
				
				MPI_Send(&num, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
				send = 0;
				
				if (limit_even == 1 && num == limit){ //last element
					break;
				}
				
			} else {
				//printf("Process %d waiting\n", my_rank);
				MPI_Recv(&local_num, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				num = local_num;
				send = 1;
				//printf("Process %d value received %d\n", my_rank, num);
			}
			
		} else {
			
			if (send == 1){
				//num = local_num;
				num++;
				printf("Process %d of %d, value = %d\n", my_rank, comm_sz, num);
				MPI_Send(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				
				send = 0;
				
				if (limit_even == 0 && num == limit){ //last element
					break;
				}
			} else {
				//printf("Process %d waiting\n", my_rank);
				MPI_Recv(&local_num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				num = local_num;
				send = 1;
				//printf("Process %d value received %d\n", my_rank, num);
			}
		}
	}
	
	MPI_Finalize();
	return 0;
}
