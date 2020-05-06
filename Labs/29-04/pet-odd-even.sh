#!/bin/bash
numbers=(1 2 4 8 16)
for i in "${numbers[@]}"; do
	echo "num of processes"
	echo $i
	mpiexec -n $i ./pet-odd-even
done
