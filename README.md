## Summary
This library translates the audio of a video from English to German, preserving the original speaker's voice and tone through voice cloning. The new audio is synchronized with the original video length and optionally lip-synced.

## Pipeline
-Load the video with MoviePy, extract the audio, and parse the SRT file<br />
-Translate each line of the SRT<br />
-Synthesize the German speech line-by-line through voice cloning<br />
-Ensure original pauses persist<br />
-Concatenate lines together into a full audio file<br />
-Synchronize the German audio to the original video by adjusting its speed<br />
-Accordingly scale the German SRT<br />
-Write the new audio onto the video<br />
-(Optionally) Lip sync the video to the new audio<br />

## Assumptions
-SRT is accurate, with no overlaps<br />
-Video has just one speaker<br />
-No cuts in the video<br />

## Limitations
-Audio stretching slightly distorts the tone<br />
-Lip-sync library produces square artifacts around the lips<br />
-Lip-sync only works on Google Colab<br />

## Setup
Set up Conda environment python==3.12
```
pip install -r req.txt
conda install ffmpeg -c conda-forge
```
Install Cuda compatible pytorch<br />


## Testing
Example test line:<br />
```
python translate.py --video_file sample_video.mp4 --transcript_srt sample_en.srt --output_dir out<br />
```
With lip-sync<br />
```
python translate.py --video_file sample_video.mp4 --transcript_srt sample_en.srt --output_dir out --do_lipsync
```
