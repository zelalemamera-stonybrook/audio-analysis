# the following script generates F0 for each soundfile identifier in the input directory, files are saved into the output directory as .csv files.

form: "Information"
	text: "source_directory", "input_wav\"
	text: "target_directory", "pitch.csv"
	real: "hertz", "75"
endform

Create Strings as file list: "fileList", source_directory$ + "*.wav"
n = Get number of strings
for i from 1 to n
	wavname$ = Get string: i
	Read from file: source_directory$ + wavname$
	iD$ = selected$ ("Sound", 1)
	writeInfoLine: source_directory$ + wavname$

	pitch = To Pitch (filtered autocorrelation): 0.0, hertz, 800.0, 15, "no", 0.03, 0.09, 0.50, 0.055, 0.35, 0.14
	matrix = To Matrix
	tableofreal = To TableOfReal
	table = To Table: "none"

	Save as comma-separated file: target_directory$ + iD$ + ".csv"
	writeInfoLine: target_directory$ + iD$ + ".csv"

	selectObject: "Sound " + iD$, pitch, matrix, tableofreal, table
	Remove
	selectObject: "Strings fileList"
endfor
Remove
