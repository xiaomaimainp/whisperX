# Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

model_id = "openai/whisper-large-v3-turbo"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

