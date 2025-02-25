import logging
import time
import os
from identify import Identify
import torch # type: ignore
import torchaudio # type: ignore
from pyannote.audio import Pipeline  # type: ignore
import gc

from transcribe import Transcribe

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


    # Create diarization pipeline
    speaker_tags = []
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token=hf_token)
    pipeline.to(device)
    waveform, sample_rate = torchaudio.load(audio)


    # Diarize file
    start_time = int(time.time())
    logger.info("running diarization...")
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, min_speakers=0, max_speakers=10)
    end_time = int(time.time())
    elapsed_time = int(end_time - start_time)
    logger.info(f"diarization done. Time taken: {elapsed_time} seconds.")

    # create a dictionary of SPEAKER_XX to real name mappings
    speaker_map = {}
    speakers = {}
    common = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):

        start = round(turn.start, 1)
        end = round(turn.end, 1)
        common.append([start, end, speaker])

        # find different speakers
        if speaker not in speaker_tags:
            speaker_tags.append(speaker)
            speaker_map[speaker] = speaker
            speakers[speaker] = []

        speakers[speaker].append([start, end, speaker])


    # identify speakers
    if voices_folder != None and voices_folder != "":
        identify = Identify(device)
        identified = []

        start_time = int(time.time())
        logger.info("running speaker recognition...")
        for spk_tag, spk_segments in speakers.items():
            spk_name = identify.speaker_recognition(audio, voices_folder, spk_segments, identified, tmp_folder)
            spk = spk_name
            identified.append(spk)
            speaker_map[spk_tag] = spk
        end_time = int(time.time())
        elapsed_time = int(end_time - start_time)
        logger.info(f"speaker recognition done. Time taken: {elapsed_time} seconds.")


    # merging same speakers
    keys_to_remove = []
    merged = []
    for spk_tag1, _ in speakers.items():
        for spk_tag2, spk_segments2 in speakers.items():
            if spk_tag1 not in merged and spk_tag2 not in merged and spk_tag1 != spk_tag2 and speaker_map[spk_tag1] == speaker_map[spk_tag2]:
                for segment in spk_segments2:
                    speakers[spk_tag1].append(segment)

                merged.append(spk_tag1)
                merged.append(spk_tag2)
                keys_to_remove.append(spk_tag2)
    
    # fixing the speaker names in common
    for segment in common:
        speaker = segment[2]
        segment[2] = speaker_map[speaker]

    for key in keys_to_remove:
        del speakers[key]
        del speaker_map[key]
    
    # transcribing the texts differently according to speaker
    transcribe = Transcribe(audio, model, language, device_name, tmp_folder)
    start_time = int(time.time())
    logger.info("running transcription...")
    for spk_tag, spk_segments in speakers.items():
        spk = speaker_map[spk_tag]
        segment_out = transcribe.segment_transcription(spk_segments)
        speakers[spk_tag] = segment_out
    end_time = int(time.time())
    elapsed_time = int(end_time - start_time)
    logger.info(f"transcription done. Time taken: {elapsed_time} seconds.")

    common_segments = []

    for item in common:
        speaker = item[2]
        start = item[0]
        end = item[1]

        for spk_tag, spk_segments in speakers.items():
            if speaker == speaker_map[spk_tag]:
                for segment in spk_segments:
                    if start == segment[0] and end == segment[1]:
                        common_segments.append([start, end, segment[2], speaker])

    return common_segments
