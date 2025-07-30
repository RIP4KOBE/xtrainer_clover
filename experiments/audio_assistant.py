"""
Audio Assistant
===========================
This script exploits OpenAI's API to create an audio assistant that can transcribe audio, generate responses.
"""

import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np
import os
from pynput import keyboard
from pydub import AudioSegment
from pydub.playback import play
import openai
from openai import OpenAI
import time
import json
import yaml


class AudioAssistant:
    def __init__(self, model, openai_api_key, base_url, user_input_filename):
        self.model = model
        self.client = OpenAI(api_key=openai_api_key, base_url=base_url)
        self.user_input_filename = user_input_filename

    def record_audio(self, sample_rate=44100, duration=5):
        print("Press 'Enter' to start recording...")
        audio_frames = np.zeros((int(sample_rate * duration), 1), dtype='float32')
        is_recording = False

        def on_press(key):
            nonlocal is_recording, audio_frames
            if key == keyboard.Key.enter and not is_recording:
                is_recording = True
                print("Recording... Release 'Enter' to stop.")
                # Start non-blocking recording
                sd.rec(samplerate=sample_rate, channels=1, dtype='float32', out=audio_frames, blocking=True)

        # define the callback function for key release
        def on_release(key):
            nonlocal is_recording
            if key == keyboard.Key.enter and is_recording:
                # Stop recording
                sd.stop()
                is_recording = False
                print("Recording stopped.")
                os.makedirs(os.path.dirname(self.user_input_filename), exist_ok=True)
                sf.write(self.user_input_filename, audio_frames, sample_rate)

                print("Audio saved as output.wav")
                return False  # Return False to stop the listener

        # Set up the listener for keyboard events
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def transcribe_audio(self):
        '''
        Transcribe the audio file using OpenAI's Whisper model
        '''
        audio_file = open(self.user_input_filename, "rb")
        transcription = self.client.audio.transcriptions.create(
            model=self.model,
            file=audio_file
        )

        print("Transcription:", transcription.text)
        return transcription.text

def play_audio(audio_file):
    sound = AudioSegment.from_mp3(audio_file)
    play(sound)

if __name__ == '__main__':

    # load the configuration from the config.json file
    with open('../configs/llm_config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # accessing configuration variables
    model= config['audio_assistant']['model']
    openai_api_key = config['api_key']
    base_url = config['base_url']
    user_input_filename = config["audio_assistant"]["user_input_filename"]

    # create an instance of the AudioAssistant class
    assistant = AudioAssistant(model, openai_api_key, base_url, user_input_filename)


    # run the audio assistant in interactive mode
    assistant.record_audio()
    transcription = assistant.transcribe_audio()
    print("Transcription:", transcription)

