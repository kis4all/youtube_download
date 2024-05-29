# 사용 예시
video_url = "https://www.youtube.com/watch?v=pzPKOhHcnvc"
output_path = "down"

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
    title = re.sub(r'[\\/:*?<>|\',]', '', title)
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


# trascribe in English using whisper

file_name = downloaded_mp3_path

import whisper
import time

start_time = time.time()

print("Mp3 is being transcribed to text file. It may take a while.")
model = whisper.load_model("tiny.en")
#model = whisper.load_model("medium.en")
result = model.transcribe(file_name, fp16 = False)

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
with open('down/' + file_name, 'r') as file:
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