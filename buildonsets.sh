#! /bin/bash
# shell script that builds a set of all valid word initial onsets observed from the TIMIT corpus

rm data_english/onsets.txt
touch data_english/onsets.txt

for dialect in $(ls TIMIT/data/TRAIN);
	do for person in $(ls TIMIT/data/TRAIN/$dialect);
		do declare -a phones
		   declare -a words
		   n=0
		   for phone in $(ls TIMIT/data/TRAIN/$dialect/$person/*.PHN);
			do phones[$n]=$phone
			   n=$(($n+1));done
		   n=0
		   for name in $(ls TIMIT/data/TRAIN/$dialect/$person/*.WRD);
			do words[$n]=$name
			   n=$(($n+1));done
		   for((i=0;i<n;i++));
			do python preprocessing_english/ExtractOnsets.py ${phones[$i]} ${words[$i]} data_english/onsets.txt;done
		   unset phones
		   unset words;done;done
