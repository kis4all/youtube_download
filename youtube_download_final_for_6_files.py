# 사용 예시 https://www.youtube.com/watch?v=JrMiSQAGOS4
video_url = "https://www.youtube.com/watch?v=w6t1oQ5G668&t=64s"
output_path = "down"

from pytubefix import YouTube
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi
import re
import datetime
import os


def download_youtube_audio(video_url, output_path):
    # 유튜브 비디오 다운로드
    yt = YouTube(video_url)
    #highest_resolution_stream = yt.streams.get_highest_resolution()
    #highest_resolution_stream.download(output_path)
    
    # 비디오에서 오디오 스트림 추출
    audio_stream = yt.streams.filter(only_audio=False).first()
    
    # 오디오 다운로드
    audio_stream.download(output_path)

    # 다운로드된 오디오 파일 경로
    title = yt.title
    #title = re.sub(r'[\\/:*?<>|\',.`‘|\(\)\[\]`\'…》\”\“\’·]', '', title)
    title = re.sub(r'[\\/:*?<>|.]', '', title)
    downloaded_audio_path = f"{output_path}/{title}.mp4"

    # mp4 파일을 mp3로 변환
    audio = AudioSegment.from_file(downloaded_audio_path, format="mp4")
    mp3_path = f"{output_path}/{title}.mp3"
    audio.export(mp3_path, format="mp3")

    # mp4 파일 삭제
    # os.remove(downloaded_audio_path)

    return mp3_path

downloaded_mp3_path = download_youtube_audio(video_url, output_path)
print(f"다운로드 및 변환 완료: {downloaded_mp3_path}")


# 자막파일 생성
file_name_srt = None

def download_auto_generated_english_subtitles(video_url):
    global file_name_srt
    try:
        video_id = video_url.split("v=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        yt = YouTube(video_url)
        title = yt.title
        title = re.sub(r'[\\/:*?<>|\',.]', '', title)
        
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
        file_name_srt = title + '.srt'
    except Exception as e:
        print(f'Error: {e}')   

download_auto_generated_english_subtitles(video_url)


# trascribe in English using whisper

file_name = downloaded_mp3_path

import whisper
import time

start_time = time.time()

print("Mp3 is being transcribed to text file. It may take a while.")
#model = whisper.load_model("tiny.en")
model = whisper.load_model("medium.en")
#result = model.transcribe(file_name, fp16 = False)
result = model.transcribe(file_name)

file_name_txt = file_name.replace("down/", "").replace(".mp3",".txt")
file_contents = result["text"]

# 파일 열기 (새 파일 생성 또는 기존 파일 덮어쓰기)
with open("down/" + file_name_txt, 'w', encoding='utf-8') as file:
    file.write(file_contents)

end_time = time.time()
excution_time = end_time - start_time

print(f"Mp3 to saved to '{file_name_txt}'")
print(f"It takes {excution_time} seconds.")



# translate into Korean using nllb

file_name = file_name_txt

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

import nltk
nltk.download('punkt')  # Download the necessary NLTK data

import time
start_time = time.time()

from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
modle = model.to(device)

# Open the text file
with open('down/' + file_name, 'r', encoding='UTF-8') as file:
    text = file.read()  # Read the entire text

# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Print each sentence
sentences_final = ""
num_list = len(sentences)
i = 0

for sentence in sentences:
    article = sentence.strip()
    #inputs = tokenizer(article, return_tensors="pt")
    inputs = tokenizer(article, return_tensors="pt").to(device)
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["kor_Hang"], max_length=30)
    
    sentence_translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    sentences_final = sentences_final + sentence_translation + " "
    
    i = i + 1
    print(f"{num_list}문장 중 {i} 문장을 번역중입니다...", end = '\r')


file_name_ko = file_name.replace(".txt","_ko.txt")
with open('down/' + file_name_ko, 'w', encoding='utf-8') as file_save:
    file_save.write(sentences_final)

end_time = time.time()
excution_time = end_time - start_time

print(f"It takes {excution_time} seconds.")
   
print(f"Translation for Korean completed and saved at '{file_name_ko}'.")


# 한국어 자막 파일 생성 파트
import pysrt
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)

file_name = file_name_srt
subs = pysrt.open('down/' + file_name, encoding='utf-8')

start_time = time.time()

for i, sub in enumerate(subs):
    inputs = tokenizer(sub.text, return_tensors="pt", truncation=True, max_length=100).to(device)
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["kor_Hang"],
        max_length=150
    )
    sub.text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"{i+1}/{len(subs)} 번역 중...", end='\r')

file_name_ko = file_name.replace(".srt", "_ko.srt")
subs.save('down/' + file_name_ko, encoding='utf-8')

end_time = time.time()
print(f"\n번역 완료! 파일 저장됨: {file_name_ko}, 시간: {end_time - start_time:.2f}초")
