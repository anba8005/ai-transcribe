import numpy as np
import logging
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)

class Diarize:
    def __init__(self, device, hf_token):
        # Create diarization pipeline
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=hf_token)
        self.pipeline.to(device)


    def process(self, audio):
        # Diarize file
        diarization = self.pipeline({"waveform": audio, "sample_rate": 16000})
        
        # create segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = round(turn.start, 1)
            end = round(turn.end, 1)
            if (end - start <= 0):
                continue
            segments.append([start, end, speaker])

        return segments



