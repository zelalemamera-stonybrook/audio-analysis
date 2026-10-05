form: "Information"
        text: "source_directory", "input_wav\"
        text: "target_directory", "formant.csv"
		real: "hertz", "162.0"
endform

Create Strings as file list: "fileList", source_directory$ + "*.wav"
n = Get number of strings
for i from 1 to n
        wavname$ = Get string: i
        Read from file: source_directory$ + wavname$
        iD$ = selected$ ("Sound", 1)
        
        writeInfoLine: source_directory$ + wavname$
        
        intensity = To Intensity: hertz, 0.0, "yes"
        matrix = Down to Matrix
        tableofreal = To TableOfReal
        table = To Table: "none"

        Save as comma-separated file: target_directory$ + iD$ + ".csv"
        writeInfoLine: target_directory$ + iD$ + ".csv"
        selectObject: "Sound " + iD$, intensity, matrix, tableofreal, table
        Remove
        selectObject: "Strings fileList"
endfor
Remove
