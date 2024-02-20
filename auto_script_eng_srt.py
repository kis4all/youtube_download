from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import datetime
import re

def download_auto_generated_english_subtitles(video_url):
    try:
        video_id = video_url.split("v=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        yt = YouTube(video_url)
        title = yt.title
        title = re.sub(r'[\\/:*?<>|]', '', title)
        
        srt_content = ""
        for i, entry in enumerate(transcript):
            start_time = str(datetime.timedelta(seconds = entry['start'])).replace('000', '')
            end_time = str(datetime.timedelta(seconds=entry['start'] + entry['duration'])).replace('000', '')   
            
            srt_content += f"{i + 1}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{entry['text']}\n\n"
            
        with open('down/' + f'{title}.srt', 'w', encoding='utf-8') as file:
            file.write(srt_content)

        print(f'Subtitles downloaded and saved as {title}.srt')
    except Exception as e:
        print(f'Error: {e}')   

        

    

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=Dbj1GNT3Mjk"
    download_auto_generated_english_subtitles(video_url)
