from youtube_transcript_api import  YouTubeTranscriptApi
import re
from pytube import YouTube

def get_auto_generated_captions(video_url):
    try:
        video_id = video_url.split("v=")[1]  # Extract video ID from the URL
        captions = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return captions
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def remove_timestamps(text):
    # Use regular expression to remove timestamps
    re.sub(r'\n', ' ', text)
    re.sub(r'\r', ' ', text)
    return re.sub(r'\[\d+:\d+\]', ' ', text)

def save_captions_to_file(captions, filename='captions.txt'):
    if captions:
        with open('down/' + filename, 'w', encoding='utf-8') as file:
            for caption in captions:
                cleaned_text = remove_timestamps(caption['text'])
                file.write(f"{cleaned_text}")
        print(f"Captions (without timestamps) saved to {filename}")
    else:
        print("Auto-generated captions not available.")

# YouTube Video URL
video_url = "https://www.youtube.com/watch?v=Dbj1GNT3Mjk"

# Get auto-generated captions
captions = get_auto_generated_captions(video_url)

# Save captions without timestamps to a file
yt = YouTube(video_url)
title = yt.title
title = re.sub(r'[\\/:*?<>|]', '', title)
print(title)
save_captions_to_file(captions, filename = title + '.txt')
