file_name = "down/West votes against rest of world in UN Human Rights Council.mp3"

import whisper
import time

start_time = time.time()

print("Mp3 is being transcribed to text file. It may take a while.")
model = whisper.load_model("medium.en")
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