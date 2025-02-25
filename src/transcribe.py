import logging
import os
from faster_whisper import WhisperModel
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class Transcribe:
    def __init__(self, file_name, model, language, device_name, tmp_folder):
        self.audio = AudioSegment.from_file(file_name, format="wav")
        self.language = language
        self.tmp_folder = tmp_folder
        self.whisper_model = WhisperModel(model, device=device_name, compute_type="float16" if device_name == "cuda" else "int8")
        

    def segment_transcription(self, segments):
        trans = ""
        texts = []

        i = 0
        for segment in segments:

            start = segment[0] * 1000   # start time in miliseconds
            end = segment[1] * 1000     # end time in miliseconds
            clip = self.audio[start:end]
            i = i + 1
            file = self.tmp_folder + "/" + "segment"+ str(i) + ".wav"
            clip.export(file, format="wav")

            try:
                trans = self.transcribe(file)  
                texts.append([segment[0], segment[1], trans])
            except Exception as err:
                print("ERROR while transcribing: ", err)

            # Delete the WAV file after processing
            os.remove(file)

        return texts
    
    def transcribe(self, file_name):
        segments, info = self.whisper_model.transcribe(file_name, language=self.language, beam_size=5)
        res = ""
        for segment in segments:
            res += segment.text + " "
        return res

    