## Summary
This library translates the audio of a video from English to German, preserving the original speaker's voice and tone through voice cloning. The new audio is synchronized with the original video length and optionally lip-synced.

## Pipeline
-Load the video with MoviePy, extract the audio, and parse the SRT file
-Translate each line of the SRT
-Synthesize the German speech line-by-line through voice cloning
-Ensure original pauses persist
-Concatenate lines together into a full audio file
-Synchronize the German audio to the original video by adjusting its speed
-Accordingly scale the German SRT
-Write the new audio onto the video
-(Optionally) Lip sync the video to the new audio

## Assumptions
-SRT is accurate, with no overlaps
-Video has just one speaker
-No cuts in the video

## Limitations
-Audio stretching distorts the tone
-Lip-sync library produces square artifacts around the lips
-Lip-sync only works on Google Colab

## Setup
Set up Conda environment python==3.12
pip install -r req.txt
conda install ffmpeg -c conda-forge
Install Cuda compatible pytorch

## Testing
Example test line:
python translate_video.py --video_file sample_video.mp4 --transcript_srt sample_en.srt --output_dir out

With lip-sync
python translate_video.py --video_file sample_video.mp4 --transcript_srt sample_en.srt --output_dir out --do_lipsync