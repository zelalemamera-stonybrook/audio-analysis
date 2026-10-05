#!/bin/bash
# the following shell script automates and implements the entire data pipeline.
# there are environmental variables and applications that are assumed to exist on the machine before running this script
# PRAAT should be set to the application location to praat. conda should be installed and have an aligner environment available for mfa. Please look at the set up
# instructions for mfa for how to add this specfically from https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html


align ()
{
	echo "aligning data..."
	python "preprocessing/PrepareAlignments.py"
	mkdir "data/alignment/source"
	cp -r "data/alignment/text/." "data/alignment/source"
	cp -r "data/alignment/audio/." "data/alignment/source"
	rm -r "data/alignment/textgrid"
	mkdir "data/alignment/textgrid"
	echo "running aligner..."
	conda run -n aligner mfa align --clean "data/alignment/source" "data/arabic_mfa.dict" "english_mfa" "data/alignment/textgrid"
	rm -r "data/alignment/source"
}

syllabify ()
{
	echo "syllabifying data..."
	python "preprocessing/GenerateIndex.py"

	rm -r "data/alignment/syllabified_audio"
	mkdir "data/alignment/syllabified_audio"

	$PRAAT --run "preprocessing/Syllabify.praat" "../data/alignment/textgrid/" "../data/syllable_index.txt" "../data/alignment/audio/" "../data/alignment/syllabified_audio/"
}

featurize ()
{
	echo "generating all features"

	rm -r "data/features"
	mkdir "data/features"

	echo "generating duration"
	mkdir "data/features/Dur"
	python "preprocessing/Dur.py" "data/alignment/syllabified_audio" "data/features/Dur"

	echo "generating Praat features"
	for name in Intensity;
		do
		mkdir "data/features/"$name"csv"
		hz=160
		while ! $PRAAT --run "preprocessing/"$name".praat" "../data/alignment/syllabified_audio/" "../data/features/"$name"csv/" $hz;do
			hz=$(($hz+1));done

		echo "transforming to tensors"
		mkdir "data/features/"$name"pt"
		python "preprocessing/Csvtopt.py" "data/features/"$name"csv" "data/features/"$name"pt"

		rm -r "data/features/"$name"csv";
		done

	for name in F0;
		do
		mkdir "data/features/"$name"csv"
		floor=75
		while ! $PRAAT --run "preprocessing/"$name".praat" "../data/alignment/syllabified_audio/" "../data/features/"$name"csv/" $floor;do
			floor=$(($floor+1));done

		echo "transforming to tensors"
		mkdir "data/features/"$name"pt"
		python "preprocessing/Csvtopt.py" "data/features/"$name"csv" "data/features/"$name"pt"

		rm -r "data/features/"$name"csv";
		done

	for name in {F1,F2,F3,F4,F5};
		do
		mkdir "data/features/"$name"csv"
		hz=160
		while ! $PRAAT --run "preprocessing/"$name".praat" "../data/alignment/syllabified_audio/" "../data/features/"$name"csv/" $hz;do
			hz=$(($hz+1));done

		echo "transforming to tensors"
		mkdir "data/features/"$name"pt"
		python "preprocessing/Csvtopt.py" "data/features/"$name"csv" "data/features/"$name"pt"

		rm -r "data/features/"$name"csv";
		done

	echo "generating melspectrogram"
	mkdir "data/features/Mel"
	$PRAAT --run "preprocessing/Mel.praat" "../data/alignment/syllabified_audio/" "../data/features/Mel/"

	echo "generating vector embeddings"
	mkdir "data/features/Wav2vecpt"
	python "preprocessing/Wav2vec.py" "data/alignment/syllabified_audio" "data/features/Wav2vecpt"

	echo "padding feature directories"
	for name in {F{0,1,2,3,4,5},Intensity,Wav2vec};
		do
		mkdir data/features/$name
		python preprocessing/Pad.py "data/features/"$name"pt" data/features/$name
		rm -r "data/features/"$name"pt";
		done

	echo "Generating Raw"
	mkdir data/features/Raw
	python "preprocessing/Padaudio.py" "data/alignment/syllabified_audio" "data/features/Raw"
}

split ()
{
	# the data lives in data/features currently. each folder in this directory contains a copy of the source data from the table.
	# batch directories have already been obtained from the table
	# by splitting it into three groups. From these tables the directory data/batch needs to  be populated.
	for batch in {train,test,dev};
		do
		rm -r data/$batch
		mkdir data/$batch;
		done
	python "preprocessing/SplitTable.py" "data/table.csv" "data"
	for batch in {train,test,dev};
		do
		for name in {Raw,Dur,F{0,1,2,3,4,5},Intensity,Wav2vec};
			do
			echo "splitting "$name" to "$batch
			rm -r "data/"$batch"/"$name
			mkdir "data/"$batch"/"$name
			python "preprocessing/Split.py" "data/features/"$name "data/"$batch"/table.csv" "data/"$batch"/"$name;
			done;
		done
}

balance ()
{
	# training data is unbalanced, this is a simple function that generates a new balanced table by directly oversampling the minority classes.
	python "preprocessing/Balance.py" "data/train/table.csv" "data/train"

}

normalize ()
{
	#normalizes the batches in question by subtracting them from the mean over the directory and dividing by the standard deviation
	for batch in {train,test,dev};
		do
		for feature in {Dur,F{0,1,2,3,4,5},Intensity};
			do
			rm -r "data/"$batch"/"$feature"norm"
			mkdir "data/"$batch"/"$feature"norm"
			python preprocessing/Normalize.py data/$batch/$feature "data/"$batch"/"$feature"norm";
			done;
		done
}

#align&syllabify&featurize&split&balance&normalize
featurize&&split&&balance&&normalize
