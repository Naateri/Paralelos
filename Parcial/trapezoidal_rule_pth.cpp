/*
Trapezoidal rule implementation using pthreads
Author: Renato Postigo
compile:
g++ -g -Wall trapezoidal_rule_pth.c -o trap_rule -std=c++11 -lpthread
execute:
./trap_rule
*/

#include <iostream>
#include <thread>

using namespace std;

pthread_mutex_t mutex;

// test function
double f(double x){
	return 3*x + 4;
}

void trap(double left, double right, int traps, double local_h, double& estimate){
	double x;
	int i;
	for(i = 1; i <= traps; i++){
		// calculate current_x
		x = left + i*local_h;
		// lock summatory
		pthread_mutex_lock(&mutex);
		// area of local trapezoid: f(x) * h
		estimate += f(x) * local_h;
		pthread_mutex_unlock(&mutex);
	}
}


int main(int argc, char *argv[]) {
	
	int num_threads;
	cout << "Threads to use: ";
	cin >> num_threads;
	
	int n = 2048, local_n;
	
	thread* threads = new thread[num_threads];
	
	double a = 0.0, b = 3.0, h, local_a, local_b;
	// total_res: global variable
	double total_res = 0.0;
	
	h = (b-a)/n;
	local_n = n/num_threads;
	// local_n: trapezoids per thread
	
	for (int i = 0; i < num_threads; i++){
		// calculate bounds for each thread
		local_a = a + i*local_n*h;
		local_b = local_a + local_n*h;
		// initialize thread
		threads[i] = thread(trap, local_a, local_b, local_n, h, ref(total_res));
	}
	
	// Wait for threads to finish
	for(int i = 0; i < num_threads; i++){
		threads[i].join();
	}
	
	cout << "Area: " << total_res << endl;
	
	delete[] threads;
	
	return 0;
}

