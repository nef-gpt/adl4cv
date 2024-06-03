# Loop through all .gif files in the current directory
  for gif_file in *.gif
          # Extract the base name of the file (without extension)
          set base_name (basename $gif_file .gif)
          
          # Construct the output file name
          set mp4_file "$base_name.mp4"
          
          # Execute the ffmpeg command to convert the gif to mp4
          ffmpeg -f gif -f lavfi -i color="121212"  -i $gif_file -vcodec h264 -pix_fmt yuv420p -n -filter_complex "[0][1]scale2ref[bg][gif];[bg]setsar=1[bg];[bg][gif]overlay=shortest=1" $mp4_fil
e
  end

# Loop through all .gif files in the current directory
for gif_file in *.mp4
        # Extract the base name of the file (without extension)
        set base_name (basename $gif_file .gif)
        
        # Construct the output file name
        set mp4_file "$base_name.cropped.mp4"
        
        # Execute the ffmpeg command to convert the gif to mp4
        ffmpeg -i $gif_file -vcodec h264 -pix_fmt yuv420p -n -vf "crop=668:1530:164:360" $mp4_file
end