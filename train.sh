#!/bin/bash
# the following shell script manages training and optimization

test ()
{
	# tests the performance of the previously trained model first argument is the name of the model, the second specifies the testing batch
	rm -r results/Model/hypothesis
	mkdir results/Model/hypothesis
	python neural_network/TestGpu.py ModelGpu results/Model/hypothesis data/dev/table.csv results/Model 0 data/dev/Praat
}


builddata ()
{
	# the praat model has a specific requirement that all of the data that currently lives in separate directories needs to now be combined into one.
	# this step can also be used to add or omit certain features in order to compare overall model performance.
	for batch in DEV;
		do
		rm -r data/$batch/Praat
		mkdir data/$batch/Praat
		python preprocessing/Praat.py data/$batch/table.csv data/$batch/Praat data/$batch/features/{Dur,F{0,1,2,3,4,5}norm,Intensitynorm};
		done
}
train ()
{

	rm -r results/$2/hypothesis
	mkdir results/$2/hypothesis

	python neural_network/UpdateDefinitions.py $4 $5 $6 $7 $8 $9 $3 ${10} ${11} # nlayers nlinear nheads outputdim seqlen featurelen batchsize mininitial maxinitial
	python neural_network/TrainGpu.py data/TRAIN/Praat data/TRAIN/balanced.csv $2 $3 1 $1

	#python neural_network/UpdateDefinitions.py 4 2 4 512 6 8 $3 -0.05 0.05
	#python neural_network/TestGpu.py $2 results/$2/hypothesis data/DEV/table.csv results/$2 0 data/DEV/Praat

	#rm -r results/$2/initial
	#mkdir results/$2/initial

	#mv results/$2/{errorlog,hypothesis_analysis,statistics}.txt results/$2/initial
	#mv results/$2/hypothesis results/$2/initial

	for ((i=0;i<0;++i));
		do
		rm -r results/$2/hypothesis
		mkdir results/$2/hypothesis

		python neural_network/UpdateDefinitions.py 3 2 4 512 6 8 $3 -0.07 0.07
		python neural_network/TrainGpu.py data/TRAIN/Praat data/TRAIN/balanced.csv $2 $3 0 $1

		#python neural_network/UpdateDefinitions.py 4 2 4 512 6 8 $3 -0.05 0.05
		#python neural_network/TestGpu.py $2 results/$2/hypothesis data/DEV/table.csv results/$2 $i data/DEV/Praat
		reset $2 $i;
		done
	echo -e '\a'
}

addmoreEpochs ()
{
	# the current model is trained on some number of epochs, this function adds the specified number of epochs to its training time for research purposes.
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

# epochs model batchsize nlayers nlinear nheads outputdim seqlen featurelen minintial maxinitial
train 100 Transformer 20 6 2 8 512 6 8 -0.05 0.05

