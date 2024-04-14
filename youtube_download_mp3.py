from pytube import YouTube
from pydub import AudioSegment
import re
import os


def download_youtube_audio(video_url, output_path):
    # 유튜브 비디오 다운로드
    yt = YouTube(video_url)
    
    # 비디오에서 오디오 스트림 추출
    audio_stream = yt.streams.filter(only_audio=True).first()
    
    # 오디오 다운로드
    audio_stream.download(output_path)

    # 다운로드된 오디오 파일 경로
    title = yt.title
    title = re.sub(r'[\\/:*?<>|\']', '', title)
    downloaded_audio_path = f"{output_path}/{title}.mp4"

    # mp4 파일을 mp3로 변환
    audio = AudioSegment.from_file(downloaded_audio_path, format="mp4")
    mp3_path = f"{output_path}/{title}.mp3"
    audio.export(mp3_path, format="mp3")

    # mp4 파일 삭제
    # os.remove(downloaded_audio_path)

    return mp3_path

# 사용 예시
video_url = "https://www.youtube.com/watch?v=VHBIfmxkHh4"
output_path = "down"

downloaded_mp3_path = download_youtube_audio(video_url, output_path)
print(f"다운로드 및 변환 완료: {downloaded_mp3_path}")
