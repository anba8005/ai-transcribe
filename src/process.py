import logging
import time
import os
from diarize import Diarize
from identify import Identify
import torch # type: ignore
import torchaudio # type: ignore
from pyannote.audio import Pipeline  # type: ignore
import gc

from transcribe import Transcribe
from vad import VAD

logger = logging.getLogger(__name__)

def process(audio, device_name, hf_token, batch_size, model, language, voices_folder, tmp_folder):

    # Print the arguments
    logger.info(f"Transcribing {audio} using model {model} on device {device_name}")
    logger.info(f"Language: {language}, Batch size: {batch_size}")
    if hf_token:
        logger.info("Using Hugging Face token for gated models.")
    if voices_folder:
        logger.info(f"Loading voice samples from {voices_folder}")


    # create device
    device = torch.device(device_name)
    if device_name == "mps":
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


    # load audio
    waveform, sample_rate = torchaudio.load(audio)
    if sample_rate != 16000:
        raise ValueError("Sample rate must be 16000")


    # perform VAD
    logger.info("running VAD...")
    start_time = int(time.time())
    #
    vad = VAD(device)
    vad_waveform = vad.process(waveform)
    #
    end_time = int(time.time())
    elapsed_time = int(end_time - start_time)
    logger.info(f"VAD done. Time taken: {elapsed_time} seconds.")
    

    # Create diarization pipeline
    logger.info("running diarization...")
    start_time = int(time.time())
    #
    diarize = Diarize(device, hf_token)
    segments = diarize.process(vad_waveform)
    #
    end_time = int(time.time())
    elapsed_time = int(end_time - start_time)
    logger.info(f"diarization done. Time taken: {elapsed_time} seconds.")


    # identify speakers
    if voices_folder != None and voices_folder != "":
        start_time = int(time.time())
        logger.info("running speaker recognition...")
        #
        identify = Identify(device, voices_folder, tmp_folder)
        identified = identify.process(audio, segments)
        #
        end_time = int(time.time())
        elapsed_time = int(end_time - start_time)
        logger.info(f"speaker recognition done. Time taken: {elapsed_time} seconds.")


    # merge consecutive segments with the same speaker
    merged = []
    if len(identified) > 0:
        current = identified[0]
        for next_segment in identified[1:]:
            # If same speaker and gap is <= 1 second
            if (current[2] == next_segment[2] and next_segment[0] - current[1] <= 1):
                # Merge by extending end time
                print('merging segments')
                current[1] = next_segment[1]
            else:
                merged.append(current)
                current = next_segment
                
        # Add the last segment
        merged.append(current)


    # transcribing the texts segment by segment
    start_time = int(time.time())
    logger.info("running transcription...")
    #
    transcribe = Transcribe(audio, model, language, device_name, tmp_folder)
    result = transcribe.segment_transcription(merged)
    #
    end_time = int(time.time())
    elapsed_time = int(end_time - start_time)
    logger.info(f"transcription done. Time taken: {elapsed_time} seconds.")

       
    return result
