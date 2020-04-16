#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>

using namespace std;

typedef vector<vector<int>> Mat;

int MAX[] = {500, 1000, 2000, 2500, 3500, 5000, 7500, 10000};
int block_size[] = {32, 64, 128, 256, 512};

Mat A, B, C;

void fill_with_zeros(Mat& A){
	int size = A.size();
	for(int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			A[i][j] = 0;
		}
	}
}

void clear_mat(Mat& A, int size){
	for(int i = 0; i < size; i++){
		A[i].clear();
	}
	A.clear();
}

void init_mat(int size, Mat& A){
	A.resize(size);
	for (int i = 0; i < size; i++){
		A[i].resize(size);
	}
	srand(time(NULL));
	for(int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			A[i][j] = rand() % 10;
		}
	}
}


void matrix_mult(Mat A, Mat B, Mat& C){ //squared matrices only
	int size = A.size();
	C.resize(size);
	for (int i = 0; i < size; i++)
		C[i].resize(size);
	int result, sum;
	for(int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			sum = 0;
			for (int k = 0; k < size; k++){
				result = A[i][k] * B[k][j];
				sum += result;
			}
			C[i][j] = sum;
		}
	}
}

void cache_matrix_mult(Mat A, Mat B, Mat& C, int block_size){
	int size = A.size();
	
	C.resize(size);
	for (int i = 0; i < size; i++)
		C[i].resize(size);
	
	fill_with_zeros(C);
	
	int result, sum;
	for(int I = 0; I < size; I += block_size){
		for (int J = 0; J < size; J += block_size){
			for (int K = 0; K < size; K += block_size){
				
				for (int i = I; i < min(I + block_size, size); i++){
					for (int j = J; j < min(J + block_size, size); j++){
						sum = 0;
						for (int k = K; k < min (K + block_size, size); k++){
							result = A[i][k] * B[k][j];
							sum += result;
						}
						C[i][j] += sum;
					}
				}
			}
		}
	}
}

int main(int argc, char *argv[]) {
	
	int cur_size, cblock_size;
	for (int i = 0; i < 7; i++){ ///Values to check on array of matrix sizes
		cur_size = MAX[i];
		init_mat(cur_size, A);
		init_mat(cur_size, B);
		cout << "Size: " << cur_size << endl;
		
		cout << "Regular algorithm\n";
		auto start = chrono::steady_clock::now();
		matrix_mult(A, B, C);
		auto end = chrono::steady_clock::now();
		
		cout << "Time " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
		
		cout << "6 blocks algorithm\n";
		for (int j = 0; j < 5; j++){ ///values to check on array of block sizes
			
			clear_mat(C, cur_size);
			
			cblock_size = block_size[j];
			if (cur_size == 500 && cblock_size == 512) cblock_size = 500;
			
			cout << "Block/Tile size " << cblock_size << endl;
			auto start1 = chrono::steady_clock::now();
			cache_matrix_mult(A, B, C, cblock_size);
			auto end1 = chrono::steady_clock::now();
			cout << "Time " << chrono::duration_cast<chrono::milliseconds>(end1 - start1).count() << endl;
			
		}
		
		clear_mat(A, cur_size);
		clear_mat(B, cur_size);
		clear_mat(C, cur_size);
	}
	
	return 0;
}

