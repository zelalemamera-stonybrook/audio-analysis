#!/bin/bash
# the following shell script manages the training and optimization of all of the models over this data. Testing results are saved in the directory under results and
# model states are saved under the specific model name in neural_network.

train ()
{
	# trains the neural network where its name is specified by the first argument. A second argument specifies the number of runs to make.
	python neural_network/TrainNetwork.py $1 data/train/Raw data/train/balanced.csv 0.001 results/$1/errorlog.txt 25 -r
	for((i=0;i<$2;++i));
		do
		python neural_network/TrainNetwork.py $1 data/train/Raw data/train/balanced.csv 0.001 results/$1/errorlog.txt 25;
		done
}

test ()
{
	# tests the performance of the previously trained model first argument is the name of the model, the second specifies the testing batch
	rm -r results/Model/hypothesis
	mkdir results/Model/hypothesis
	python neural_network/Test.py Model results/Model/hypothesis data/dev/table.csv results/Model 0 data/dev/{Intensitynorm,F{0,1,2,3,4,5}norm,Dur}
}


buildpraat ()
{
	# the praat model has a specific requirement that all of the data that currently lives in separate directories needs to now be combined into one.
	# this step can also be used to add or omit certain features in order to compare overall model performance.
	for batch in {train,test,dev};
		do
		rm -r data/$batch/Praat
		mkdir data/$batch/Praat
		python preprocessing/Praat.py data/$batch/table.csv data/$batch/Praat data/$batch/{Dur,F{0,1,2,3,4,5}norm,Intensitynorm};
		done
}
trainpraat ()
{
	python neural_network/Train.py Model data/train/balanced.csv results/Model/errorlog.txt 25 -r data/train/{Intensitynorm,F{0,1,2,3,4,5}norm,Dur}
	rm -r results/Model/hypothesis
	mkdir results/Model/hypothesis
	python neural_network/Test.py Model results/Model/hypothesis data/dev/table.csv results/Model 0 data/dev/{Intensitynorm,F{0,1,2,3,4,5}norm,Dur}
	rm -r results/Model/initial
	mkdir results/Model/initial
	mv results/Model/{errorlog,hypothesis_analysis,statistics}.txt results/Model/initial
	mv results/Model/hypothesis results/Model/initial

	for ((i=0;i<$1;++i));
		do
		python neural_network/Train.py Model data/train/balanced.csv results/Model/errorlog.txt 25 data/train/{Intensitynorm,F{0,1,2,3,4,5}norm,Dur}
		rm -r results/Model/hypothesis
		mkdir results/Model/hypothesis
		python neural_network/Test.py Model results/Model/hypothesis data/dev/table.csv results/Model $i data/dev/{Intensitynorm,F{0,1,2,3,4,5}norm,Dur}
		reset Model $i;
		done
	echo -e '\a'
}

reset ()
{
	# helper code that resets the training session for a model so that it can be retrained
	rm -r results/$1/"session"$2
	mkdir results/$1/"session"$2
	cp results/$1/{errorlog.txt,hypothesis_analysis.txt,statistics.txt} results/$1/"session"$2
	mkdir results/$1/"session"$2/hypothesis
	cp results/$1/hypothesis/* results/$1/"session"$2/hypothesis

}


#buildpraat
trainpraat 100
#test
#train BaselineModel 100
