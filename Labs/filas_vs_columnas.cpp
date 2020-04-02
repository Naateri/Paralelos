#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <vector>

using namespace std;

int MAX[] = {1000, 2000, 5000, 10000, 20000, 22500};

int y;

vector<vector<int>> A;
vector<int> X;

void init_A(int size){
	A.resize(size);
	for (int i = 0; i < size; i++){
		A[i].resize(size);
	}
	srand(time(NULL));
	for(int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			A[i][j] = rand() % 1000000;
		}
	}
}
	
void init_X(int size){
	
	X.resize(size);
	
	srand(time(NULL));
	for(int i = 0; i < size; i++){
		X[i] = rand() % 1000000;
	}
}
	
void loop_one(int size){
	//y = 0 before loop
	for(int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			y += A[i][j] * X[j];
		}
	}
}
	
void loop_two(int size){
	//y = 0 before loop
	for(int j = 0; j < size; j++){
		for (int i = 0; i < size; i++){
			y += A[i][j] * X[j];
		}
	}
}

int main(int argc, char *argv[]) {
	int cur_size;
	for (int i = 0; i < 6; i++){
		cur_size = MAX[i];
		init_A(cur_size);
		init_X(cur_size);
		cout << "Size: " << cur_size << endl;
		
		y = 0;
		auto start = chrono::steady_clock::now();
		loop_one(cur_size);
		auto end = chrono::steady_clock::now();
		
		cout << "Loop one " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
		
		y = 0;
		auto start1 = chrono::steady_clock::now();
		loop_two(cur_size);
		auto end1 = chrono::steady_clock::now();
		
		cout << "Loop two " << chrono::duration_cast<chrono::milliseconds>(end1 - start1).count() << endl;
		
		for (int j = 0; j < cur_size; j++){
			A[j].clear();
		}
		A.clear();
		X.clear();
	}
	
	return 0;
}

