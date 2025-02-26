import logging
from speechbrain.inference import SpeakerRecognition # type: ignore
import os
import os
from collections import defaultdict
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class Identify:
    def __init__(self, device, voices_folder, tmp_folder):
        self.verification = SpeakerRecognition.from_hparams(run_opts={"device":device}, source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
        self.voices_folder = voices_folder
        self.tmp_folder = tmp_folder

    def process(self, file_name, segments):
        speakers = os.listdir(self.voices_folder)
        identified = []

        # Load the WAV file
        audio = AudioSegment.from_file(file_name, format="wav")
        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)

        i = 0
        for segment in segments:
            start = segment[0] * 1000   # start time in miliseconds
            end = segment[1] * 1000     # end time in miliseconds
            if (end - start <= 0):
                continue
            clip = audio[start:end]
            i = i + 1
            file = self.tmp_folder + "/" + file_name.split("/")[-1].split(".")[0] + "_segment"+ str(i) + ".wav"
            clip.export(file, format="wav")

            max_score = 0
            person = "unknown"      # if no match to any voice, then return unknown

            for speaker in speakers:

                voices = os.listdir(self.voices_folder + "/" + speaker)

                for voice in voices:
                    voice_file = self.voices_folder + "/" + speaker + "/" + voice

                    try:
                        # compare voice file with audio file
                        score, prediction = self.verification.verify_files(voice_file, file)
                        prediction = prediction[0].item()
                        score = score[0].item()

                        if prediction == True and score >= max_score:
                            max_score = score
                            person = speaker.split(".")[0]  
                    except Exception as err:
                        logger.error("error occured while speaker %s in file %s between %s and %s recognition: %s", speaker, file, start, end, err)
                        break

            # Delete the WAV file after processing
            os.remove(file)

            # add person to identified
            identified.append([segment[0], segment[1], person])

        return identified