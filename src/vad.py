import torch 
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VAD:
    def __init__(self, device, chunk_size=30, vad_onset=0.500, vad_offset=0.363):
         # Load the model
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False,
                                  verbose=False)
        self.device = device
        self.model = self.model.to(device)
        self.chunk_size = chunk_size
        self.vad_onset = vad_onset
        self.vad_offset = vad_offset


    def process(self, audio):

        # Get speech timestamps
        vad_segments = self.get_speech_timestamps(audio)
        
        # Create a new audio tensor with the same shape as the original audio
        vad_audio = torch.zeros_like(audio)
        
        # Copy only the VAD segments from original audio
        for segment in vad_segments:
            chunk_start = segment["start"]
            chunk_end = segment["end"]
            vad_audio[:, chunk_start:chunk_end] = audio[:, chunk_start:chunk_end]

        # Return tensor with correct shape (channel, time)
        return vad_audio
        

    def get_speech_timestamps(self, audio):
        get_speech_timestamps, *_ = self.utils
        
        # Ensure audio is on the same device as the model
        if isinstance(audio, torch.Tensor):
            audio = audio.to(self.device)
            
        return get_speech_timestamps(
            audio, 
            self.model, 
            sampling_rate=16000,
            threshold=0.8,              # Higher threshold = more conservative detection
            min_speech_duration_ms=250, # Minimum speech segment duration in ms
            min_silence_duration_ms=100 # Minimum silence segment duration in ms
        )
