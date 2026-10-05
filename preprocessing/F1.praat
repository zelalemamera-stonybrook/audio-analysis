# The following script generates formant data on all of the wave files identified in the source directory. Data is written as a csv file for each wavefile, and saved with the same name in the target directory.

form: "Information"
	text: "source_directory", "input_wav\"
	text: "target_directory", "formant.csv"
	real: "hertz", "5"
endform

Create Strings as file list: "fileList", source_directory$ + "*.wav"
n = Get number of strings
for i from 1 to n
	wavname$ = Get string: i
	Read from file: source_directory$ + wavname$
	iD$ = selected$ ("Sound", 1)
	
	writeInfoLine: source_directory$ + wavname$
	
	formant = To Formant (burg): 0.0, 5, 5500.0, 0.025, 50.0
	matrix = To Matrix: 1
	tableofreal = To TableOfReal
	table = To Table: "none"

	Save as comma-separated file: target_directory$ + iD$ + ".csv"
	writeInfoLine: target_directory$ + iD$ + ".csv"
	selectObject: "Sound " + iD$, formant, matrix, tableofreal, table
	Remove
	selectObject: "Strings fileList"
endfor
Remove
	
