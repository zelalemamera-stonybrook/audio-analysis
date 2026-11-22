#! /bin/bash

#Shell script for building words out of a sentence and then syllabifying the english audio corpus

makeSentences ()
{
rm -r data_english/$1
mkdir data_english/$1
mkdir data_english/$1/sentence
mkdir data_english/$1/sentence/{audio,text}
i=0

for dialect in DR{1,2,3,4,5,6,7,8};do
for person in $(ls TIMIT/data/$1/$dialect);
	do for sentence in $(ls TIMIT/data/$1/$dialect/$person/*.TXT);
		do cp $sentence data_english/$1/sentence/text/$i.txt
		   i=$(($i+1));done;done;
done

i=0

for dialect in DR{1,2,3,4,5,6,7,8};do
for person in $(ls TIMIT/data/$1/$dialect);
	do for audio in $(ls TIMIT/data/$1/$dialect/$person/*.WAV);
		do cp $audio data_english/$1/sentence/audio/$i.wav
	   	   i=$(($i+1));done;done;
done

}

makeWords ()
{
rm -r data_english/$1/word
mkdir data_english/$1/word&&
mkdir data_english/$1/word/{audio,text}&&
i=0&&
for dialect in DR{1,2,3,4,5,6,7,8};do
for person in $(ls TIMIT/data/$1/$dialect);
	do for words in $(ls TIMIT/data/$1/$dialect/$person/*.WRD);
		do mkdir data_english/$1/word/audio/$i&&
		   python preprocessing_english/breakup.py data_english/$1/sentence/audio/$i.wav $words data_english/$1/word/audio/$i&&
		   cp $words data_english/$1/word/text/$i.txt&&
		   i=$(($i+1));done;done;
done

}

makeSyllables ()
{
rm -r data_english/$1/syllable
mkdir data_english/$1/syllable
mkdir data_english/$1/syllable/{audio,text,gold}
python preprocessing_english/GenerateLexiconDictionary.py TIMIT/data/DOC/TIMITDIC.TXT "TIMITDIC.txt"
i=0
for dialect in DR{1,2,3,4,5,6,7,8};do
for person in $(ls TIMIT/data/$1/$dialect);
	do for phones in $(ls TIMIT/data/$1/$dialect/$person/*.PHN);
		do mkdir data_english/$1/syllable/{audio,gold}/$i
		   j=0
		   for word in $(ls data_english/$1/word/audio/$i);
			do mkdir data_english/$1/syllable/{audio,gold}/$i/$j
			   j=$(($j+1));done
		   cp $phones data_english/$1/syllable/text/$i.txt
		   python preprocessing_english/SyllabifySentence.py data_english/$1/sentence/audio/$i.wav $phones data_english/$1/word/text/$i.txt data_english/$1/syllable/audio/$i
		   python preprocessing_english/LabelSentence.py data_english/$1/syllable/audio/$i data_english/$1/word/text/$i.txt TIMITDIC.txt data_english/$1/syllable/gold/$i $i
		   i=$(($i+1));
		   done;done;
done
}

joinPipeline ()
{
# Data is currently disjoint from the main pipeline. This script transforms the result of the previous process to match the setup required at the beginning of
# the featurization step of the main pipeline. This involves doing two things: collecting all of the words from english, and storing their syllabified audio vecors into
# the syllabified audio directory. Words with only one syllable will be discarded. And secondly, a table containing descriptions of each datapoint must be stored at data/table.csv
rm -r data/$1
mkdir data/$1
mkdir data/$1/syllabified_audio
rm data/$1/table.csv

i=0
for sentence in $(ls data_english/$1/syllable/audio);
	do for word in $(ls data_english/$1/syllable/audio/$sentence);
		do n=0
		   if [[ $i -eq 1400 ]];then
			ls data_english/$1/syllable/{text,audio}/$sentence/$word >> edgecase.txt
			echo sentence $sentence word $word >> edgecase.txt;fi
		   for syllable in $(ls data_english/$1/syllable/gold/$sentence/$word);
			do n=$(($n+1));done
		   if [[ $n -ge 2 ]] && [[ $n -le 10 ]];then
			for syllable in $(ls data_english/$1/syllable/audio/$sentence/$word);
				do cp "data_english/"$1"/syllable/audio/"$sentence"/"$word"/"$syllable "data/"$1"/syllabified_audio/"$i"_"$syllable;done
			python preprocessing_english/JoinGold.py data_english/$1/syllable/gold/$sentence/$word data_english/$1/word/text/$sentence.txt $word $sentence TIMITDIC.txt $i data/$1/table.csv
			i=$(($i+1));fi;
		   done;
	   done;
}




#makeSentences "TEST"
#makeWords "TEST"
#makeSyllables "TEST"
joinPipeline "TEST"
