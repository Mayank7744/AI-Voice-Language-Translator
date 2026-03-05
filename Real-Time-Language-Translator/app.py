import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs.client import ElevenLabs
import uuid
from pathlib import Path
import os


def audio_transcription(audio_file):

    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_file)

    return transcription


def text_translation(text):

    translator_es = Translator(from_lang="en", to_lang="es")
    translator_tr = Translator(from_lang="en", to_lang="tr")
    translator_ja = Translator(from_lang="en", to_lang="ja")

    es_text = translator_es.translate(text)
    tr_text = translator_tr.translate(text)
    ja_text = translator_ja.translate(text)

    return es_text, tr_text, ja_text


def text_to_speech(text):

    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    response = client.text_to_speech.convert(
        voice_id="pMsXgVXv3BLzUgSXRplE",
        text=text,
        model_id="eleven_multilingual_v2"
    )

    save_file_path = f"{uuid.uuid4()}.mp3"

    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    return save_file_path


def voice_to_voice(audio_file):

    transcription = audio_transcription(audio_file)

    text = transcription.text

    es, tr, ja = text_translation(text)

    es_audio = text_to_speech(es)
    tr_audio = text_to_speech(tr)
    ja_audio = text_to_speech(ja)

    return Path(es_audio), Path(tr_audio), Path(ja_audio)


audio_input = gr.Audio(
    sources=["microphone"],
    type="filepath"
)

demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[
        gr.Audio(label="Spanish"),
        gr.Audio(label="Turkish"),
        gr.Audio(label="Japanese")
    ],
    title="Real-Time Language Translator"
)

if __name__ == "__main__":
    demo.launch()