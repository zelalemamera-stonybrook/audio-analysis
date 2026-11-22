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

	rm -r "data/"$1"/features"
	mkdir "data/"$1"/features"

	echo "generating duration"
	mkdir "data/"$1"/features/Dur"
	python "preprocessing/Dur.py" "data/"$1"/syllabified_audio" "data/"$1"/features/Dur"

	echo "generating Praat features"
	for name in Intensity;
		do
		mkdir "data/"$1"/features/"$name"csv"
		hz=160
		while ! $PRAAT --run "preprocessing/"$name".praat" "../data/"$1"/syllabified_audio/" "../data/"$1"/features/"$name"csv/" $hz;do
			hz=$(($hz+1));done

		echo "transforming to tensors"
		mkdir "data/"$1"/features/"$name"pt"
		python "preprocessing/Csvtopt.py" "data/"$1"/features/"$name"csv" "data/"$1"/features/"$name"pt"

		rm -r "data/"$1"/features/"$name"csv";
		done

	for name in F0;
		do
		mkdir "data/"$1"/features/"$name"csv"
		floor=75
		while ! $PRAAT --run "preprocessing/"$name".praat" "../data/"$1"/syllabified_audio/" "../data/"$1"/features/"$name"csv/" $floor;do
			floor=$(($floor+1));done

		echo "transforming to tensors"
		mkdir "data/"$1"/features/"$name"pt"
		python "preprocessing/Csvtopt.py" "data/"$1"/features/"$name"csv" "data/"$1"/features/"$name"pt"

		rm -r "data/"$1"/features/"$name"csv";
		done

	for name in {F1,F2,F3,F4,F5};
		do
		mkdir "data/"$1"/features/"$name"csv"
		hz=160
		while ! $PRAAT --run "preprocessing/"$name".praat" "../data/"$1"/syllabified_audio/" "../data/"$1"/features/"$name"csv/" $hz;do
			hz=$(($hz+1));done

		echo "transforming to tensors"
		mkdir "data/"$1"/features/"$name"pt"
		python "preprocessing/Csvtopt.py" "data/"$1"/features/"$name"csv" "data/"$1"/features/"$name"pt"

		rm -r "data/"$1"/features/"$name"csv";
		done
}

mel ()
{
	echo "generating melspectrogram"
	mkdir "data/features/Mel"
	$PRAAT --run "preprocessing/Mel.praat" "../data/alignment/"$1"/" "../data/features/Mel/"
}

wav2vec ()
{
	echo "generating vector embeddings"
	rm -r "data/"$1"/features/Wav2vecpt"
	mkdir "data/"$1"/features/Wav2vecpt"
	python "preprocessing/Wav2vec.py" "data/"$1"/syllabified_audio" "data/"$1"/features/Wav2vecpt"

}

padwav2vec ()
{
	rm -r data/$1/features/wav2vec
	mkdir data/$1/features/wav2vec
	python preprocessing/Pad.py data/$1/features/Wav2vecpt data/$1/features/wav2vec
	rm -r data/$1/features/Wav2vecpt
}

splitwav2vec ()
{
	for batch in {train,test,dev};
		do
			echo "splitting "wav2vec" to "$batch
			rm -r "data/"$batch"/"wav2vec
			mkdir "data/"$batch"/"wav2vec
			python "preprocessing/Split.py" "data/features/"wav2vec "data/"$batch"/table.csv" "data/"$batch"/"wav2vec;
		done

}

pad ()
{
	echo "padding feature directories"
	for name in {F{0,1,2,3,4,5},Intensity};
		do
		mkdir data/$1/features/$name
		python preprocessing/Pad.py "data/"$1"/features/"$name"pt" data/$1/features/$name
		rm -r "data/"$1"/features/"$name"pt";
		done

	echo "Generating Raw"
	mkdir data/$1/features/Raw
	python "preprocessing/Padaudio.py" "data/"$1"/syllabified_audio" "data/"$1"/features/Raw"
}

split ()
{
# moves a determined subset of the provided directory and saves it to a new directory by the provided name
	SEED=3529145006359120161

	rm -r data/$2
	mkdir data/$2
	mkdir data/$2/features

	for name in {Raw,Durnorm,F{0,1,2,3,4,5}norm,Intensitynorm};
		do
		mkdir "data/"$2"/features/"$name;
		done;
	for id in $(python preprocessing/Extract.py data/$1/table.csv 0.5 data/$2/table.csv $SEED);
		do for name in {Raw,Dur,F{0,1,2,3,4,5}norm,Intensitynorm};
			do
			echo moving $1/$name/$id to $2/$name/$id
			mv "data/"$1"/features/"$name"/"$id"_"*".pt" data/$2/features/$name/;
			done;
		done
}

balance ()
{
	# training data is unbalanced, this is a simple function that generates a new balanced table by directly oversampling the minority classes.
	python "preprocessing/Balance.py" "data/"$1"/table.csv" "data/"$1

}

normalize ()
{
	#normalizes the batches in question by subtracting them from the mean over the directory and dividing by the standard deviation
	for batch in $1;
		do
		for feature in {Dur,F{0,1,2,3,4,5},Intensity};
			do
			rm -r "data/"$batch"/features/"$feature"norm"
			mkdir "data/"$batch"/features/"$feature"norm"
			rm -r "data/"$batch"/features/"$feature"speakernorm"
			mkdir "data/"$batch"/features/"$feature"speakernorm"
			python preprocessing/NormalizeSpeakers.py data/$batch/features/$feature data/$batch/table.csv "data/"$batch"/features/"$feature"speakernorm"
			python preprocessing/Normalize.py "data/"$batch"/features/"$feature"speakernorm" "data/"$batch"/features/"$feature"norm";
			done;
		done
}

#featurize TEST
#wav2vec TEST
#padwav2vec TEST
#pad TEST
#balance TRAIN
#normalize TEST
#split TEST DEV
