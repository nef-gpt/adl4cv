#!/bin/bash

# Define the input MP4 file and output directory
input_file="./unconditioned_35_cropped.mp4"
output_dir="./unconditioned_35_cropped_pics"

transparent_color="#101010"


# Create the output directory if it doesn't exist, or clear it if it does
if [ -d "$output_dir" ]; then
  rm -rf "$output_dir/*"
else
  mkdir -p "$output_dir"
fi

# Extract frames from the input video and save them as PNG files in the output directory
ffmpeg -i "$input_file" "$output_dir/frame_%04d.png"


# Convert specific color to transparent in each frame using ImageMagick
for frame in "$output_dir"/*.png; do
  convert "$frame" -transparent "$transparent_color" "$frame"
done
