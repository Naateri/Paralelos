#!/bin/bash
threads=(1 2 4)

for i in "${threads[@]}"; do
	echo "num of threads"
	echo $i
	echo "8000000x8"
	./mat_vect $i 8000000 8
	echo " "
	
	echo "8000x8000"
	./mat_vect $i 8000 8000
	echo " "
	
	echo "8x8000000"
	./mat_vect $i 8 8000000
	echo " "
done

