#!/bin/bash

pyt preprocessing_english/SyllabifySentence.py data_english/Train/sentence/audio/181.wav data_english/Train/syllable/text/181.txt data_english/Train/word/text/181.txt data_english/Train/syllable/audio/181
pyt preprocessing_english/LabelSentence.py data_english/Train/syllable/audio/181 data_english/Train/word/text/181.txt TIMITDIC.txt data_english/Train/syllable/gold/181 181
