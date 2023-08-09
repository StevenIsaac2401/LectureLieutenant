import os
import whisper
from transformers import T5ForConditionalGeneration, T5Tokenizer
from decouple import config

lecture_path = config('LECTURE_PATH')

# transcribe audio
def transcribe(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

# summarize transcript given text and max length
def summarize(full_text, max_length):
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    input_text = "summarize: " + full_text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    summary = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(summary[0], skip_special_tokens=False)

# append to a file with path specified in .env file
def write_to_folder(text):
    file1 = open(lecture_path, "a")  # append mode
    file1.write(text + '\n')
    file1.close()

def main():
    audio_file = "the-gettysburg-address.mp3"
    transcript = transcribe(audio_file)
    summary = summarize(transcript, 20) # parameters: text, max length of summary
    print(summary)
    write_to_folder(summary)

if __name__ == "__main__":
    main()