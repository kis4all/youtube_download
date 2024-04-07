file_name = 'down/West votes against rest of world in UN Human Rights Council.txt'


import nltk
#nltk.download('punkt')  # Download the necessary NLTK data

import time
start_time = time.time()

from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# Open the text file
with open(file_name, 'r') as file:
    text = file.read()  # Read the entire text

# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Print each sentence
sentences_final = ""
num_list = len(sentences)
i = 0

for sentence in sentences:
    article = sentence.strip()
    inputs = tokenizer(article, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["kor_Hang"], max_length=30)
    
    sentence_translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    sentences_final = sentences_final + sentence_translation + " "
    
    i = i + 1
    print(f"{num_list}문장 중 {i} 문장을 번역중입니다...", end = '\r')


file_name_ko = file_name.replace(".txt","_ko.txt")
with open(file_name_ko, 'w', encoding='utf-8') as file_save:
    file_save.write(sentences_final)

end_time = time.time()
excution_time = end_time - start_time

print(f"It takes {excution_time} seconds.")
   
print(f"Translation for Korean completed and saved at '{file_name_ko}'.")






