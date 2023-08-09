import whisper
from transformers import T5ForConditionalGeneration, T5Tokenizer

def transcribe(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

def summarize(full_text, max_length):
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    input_text = "summarize: " + full_text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    summary = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(summary[0], skip_special_tokens=False)

def main():
    transcript = transcribe("the-gettysburg-address.mp3")
    summary = summarize(transcript, 20) # parameters: text, max length of summary
    print(summary)

if __name__ == "__main__":
    main()