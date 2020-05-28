#!/bin/bash
threads=(1 2 3 4 8 16 32)

echo "250k elements"
for i in "${threads[@]}"; do
	echo "num of threads"
	echo $i
	
	echo "Two parallel for"
	./odd_even_parfor $i 250000
	echo " "
	
	echo "Two for directives"
	./odd_even_fordir $i 250000
	echo " "
done

