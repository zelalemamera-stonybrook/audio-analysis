#! /bin/bash

#Shell script for building words out of a sentence audio corpus and intervals provided as text

makeWords ()
{
rm -r data_english/Train/word
mkdir data_english/Train/word&&
mkdir data_english/Train/word/{audio,text}&&
i=0&&
for person in $(ls TIMIT/data/TRAIN/DR6);
	do for words in $(ls TIMIT/data/TRAIN/DR6/$person/*.WRD);
		do mkdir data_english/Train/word/audio/$i&&
		   python preprocessing_english/breakup.py data_english/Train/sentence/audio/$i.pt $words data_english/Train/word/audio/$i&&
		   cp $words data_english/Train/word/text/$i.txt&&
		   i=$(($i+1));done;done

}

makeSyllables ()
{
rm -r data_english/Train/syllable
mkdir data_english/Train/syllable
mkdir data_english/Train/syllable/{audio,text,gold}
i=0
for person in $(ls TIMIT/data/TRAIN/DR6);
	do for phones in $(ls TIMIT/data/TRAIN/DR6/$person/*.PHN);
		do mkdir data_english/Train/syllable/$i
		   j=0
		   for word in $(ls data_english/Train/word/audio/$i);
			do mkdir data_english/Train/syllable/audio/$i/$j
			   mkdir data_english/Train/syllable/gold/$i/$j
			   j=$(($j+1));done
		   cp phones data_english/Train/syllable/text/$i.txt
		   python preprocessing_english/SyllabifyAndLabelSentence.py data_english/Train/word/audio/$i data_english/Train/word/text/$i.txt TIMIT/data/DOC/TIMITDIC.TXT phones data_english/Train/syllable/audio/$i data_english/Train/syllable/gold/$i
		   i=$(($i+1));done;done

}
