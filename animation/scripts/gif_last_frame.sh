#!/bin/bash

# Define the input gif file
gif_file="../submissions/presentation_1/public/comparison_0/unconditioned_0.gif"

# Extract the base name of the file (without extension)
base_name=$(basename "$gif_file" .gif)

# Define the output file name
png_file="../submissions/presentation_1/public/comparison_0/${base_name}_last_frame.png"

# Execute the ffmpeg command to extract the last frame of the gif and crop the image
ffmpeg -i "$gif_file" -vf "select=eq(n\,$(($(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "$gif_file")-1))),crop=668:1530:164:360" -vsync vfr "$png_file"

echo "Last frame has been saved and cropped to: $png_file"
