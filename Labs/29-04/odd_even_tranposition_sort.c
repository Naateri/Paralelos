#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int get_random(int max){
	srand( (unsigned) time(0));
	return rand() % max;
}

int compare(const void* a, const void* b){
	return ( *(int*)a > *(int*) b );
}

int* generate_array(int size){
	int* array = (int*) (malloc(size*sizeof(int));
	for (int i = 0; i < size; i++){
		array[i] = get_random(max);
	}
	return array;
}

void Merge_low(int my_keys[], int recv_keys[], int temp_keys[], int local_n){
	int m_i, r_i, t_i;
	
	m_i = r_i = r_i = 0;
	while (t_i < local_n){
		if (my_keys[m_i] <= recv_keys[r_i]){
			temp_keys[t_i] = my_keys[t_i];
			t_i++; m_i++;
		} else {
			temp_keys[t_i] = recv_keys[r_i];
			t_i++; r_i++;
		}
	}
	
	for (m_i = 0; m_i < local_n; m_i++){
		my_keys[m_i] = temp_keys[m_i];
	}
}

int keys[] = {100000, 200000, 400000, 800000};

int main(int argc, char *argv[]) {
	
	int my_rank, comm_sz;
	
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	
	int* array;
	
	for(int i = 0; i < 4; i++){
		if (my_rank == 0) array = generate_array(keys[i]);
		else array = (int*) malloc(keys[i]*sizeof(int));
		MPI_Bcast(array, keys[i], MPI_INT, 0, MPI_COMM_WORLD);
	}
	
	return 0;
}

