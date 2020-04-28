#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(){
	int comm_sz, my_rank, local_limit, limit, num = 0, local_num = 0;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	if (my_rank == 0){
		scanf("%d", &limit);
		printf("limit is %d\n", limit);
		MPI_Send(&limit, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
	} else {
		MPI_Recv(&limit, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("Process %d, limit %d\n", my_rank, limit);
		local_num = 1;
	}
		

	while (num < limit){
		printf("num %d\n", num);
		if (my_rank == 0){
			
			num = local_num;
			num++;
			printf("Process %d of %d, value = %d", my_rank, comm_sz, num);
			
			MPI_Send(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
			
			MPI_Recv(&local_num, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			MPI_Recv(&local_num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			num = local_num;
			num++;
			printf("Process %d of %d, value = %d", my_rank, comm_sz, num);
			MPI_Send(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
	}
	
	MPI_Finalize();
	return 0;
}
