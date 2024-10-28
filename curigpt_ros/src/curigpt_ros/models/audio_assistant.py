"""
Audio Assistant for CuriGPT
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


class AudioAssistant:
    def __init__(self, openai_api_key, base_url, user_input_filename, curigpt_output_filename):
        self.client = OpenAI(api_key=openai_api_key, base_url=base_url)
        self.user_input_filename = user_input_filename
        self.curigpt_output_filename = curigpt_output_filename

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
            # else:
            #     # print("Wait for recording...")

        # define the callback function for key release
        def on_release(key):
            nonlocal is_recording
            if key == keyboard.Key.enter and is_recording:
                # Stop recording
                sd.stop()
                is_recording = False
                print("Recording stopped.")
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
            model="whisper-1",
            file=audio_file
        )

        print("Transcription:", transcription.text)
        return transcription.text

    def generate_response(self, text):
        '''
        Generate a response using OpenAI's GPT-3.5 model
        '''
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}]
        )
        print(response)
        return response.choices[0].message.content.strip()


    def text_to_speech(self, text):
        '''
        Convert text to speech using OpenAI's TTS model
        '''
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )

        response.stream_to_file(self.curigpt_output_filename)
        sound = AudioSegment.from_mp3(self.curigpt_output_filename)
        play(sound)

    def audio_demo(self):
        '''
        Play the demo audio files
        '''
        demo1_audio = "assets/chat_audio/bicchi_demo/bicchi_demo1.mp3"
        demo2_audio = "assets/chat_audio/bicchi_demo/bicchi_demo2.mp3"
        demo3_audio = "assets/chat_audio/bicchi_demo/bicchi_demo3.mp3"
        demo4_audio = "assets/chat_audio/bicchi_demo/bicchi_demo4.mp3"

        audio_files = [demo1_audio, demo2_audio, demo3_audio, demo4_audio]

        for audio_file in audio_files:
            # print("Playing:", audio_file)
            time.sleep(8)
            sound = AudioSegment.from_mp3(audio_file)
            play(sound)

    def audio_demo_chinese(self):
        '''
        Play the demo audio files
        '''
        demo1_audio = "assets/chat_audio/chinese_demo/demo1.mp3"
        demo2_audio = "assets/chat_audio/chinese_demo/demo2.mp3"
        demo3_audio = "assets/chat_audio/chinese_demo/demo3.mp3"
        demo4_audio = "assets/chat_audio/chinese_demo/demo4.mp3"

        audio_files = [demo1_audio, demo2_audio, demo3_audio, demo4_audio]

        for audio_file in audio_files:
            # print("Playing:", audio_file)
            time.sleep(8)
            sound = AudioSegment.from_mp3(audio_file)
            play(sound)


def play_audio(audio_file):
    sound = AudioSegment.from_mp3(audio_file)
    play(sound)

if __name__ == '__main__':

    # load the configuration from the config.json file
    with open('config/config.json', 'r') as config_file:
        config = json.load(config_file)

    # accessing configuration variables
    openai_api_key = config['openai_api_key']
    base_url = config['base_url']
    user_input_filename = config['user_input_filename']
    # user_input_filename = "assets/chat_audio/bicchi_demo/bicchi_demo3.mp3"
    # curigpt_output_filename = config['curigpt_output_filename']
    curigpt_output_filename = "assets/chat_audio/welcome_audio/demo_ending.mp3"


    # create an instance of the AudioAssistant class
    assistant = AudioAssistant(openai_api_key, base_url, user_input_filename, curigpt_output_filename)

    # audio2text or text2audio
    # assistant.transcribe_audio()
    assistant.text_to_speech("You're welcome! If you have any more questions or need further assistance, feel free to ask. Have a great day!")


    # audio demo
    # assistant.audio_demo()
    # assistant.audio_demo_chinese()


    # run the audio assistant in interactive mode
    # assistant.record_audio()
    # transcription = assistant.transcribe_audio()
    # response_text = assistant.generate_response(transcription)
    # assistant.text_to_speech(response_text)

