import argparse
import torch # type: ignore
import logging
from process import process  # Import the transcribe function

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logging.getLogger('speechbrain.utils.fetching').setLevel(logging.WARNING)
logging.getLogger('speechbrain.utils.quirks').setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", type=str, help="audio file to transcribe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Access Token to access PyAnnote gated models")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for PyTorch inference")
    parser.add_argument("--model", default="large-v3", help="name of the Whisper model to use")
    parser.add_argument("--language", default="lt", help="language to transcribe")
    parser.add_argument("--voices_folder",   default=None, help="folder to load voice samples from")
    parser.add_argument("--tmp_folder", default="/tmp", help="folder to save temporary files")

    args = parser.parse_args()
    
    # Call the process function with all arguments
    segments = process(
        audio=args.audio,
        device_name=args.device,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        model=args.model,
        language=args.language,
        voices_folder=args.voices_folder,
        tmp_folder=args.tmp_folder
    )

    # print the result
    entry = ""
    for segment in segments:
        start = segment[0]
        end = segment[1]
        text = segment[2]
        speaker = segment[3]
        
        if text != "" and text != None:
            entry += f"{speaker} ({start} : {end}) : {text}\n"

    print(entry)

if __name__ == "__main__":
    main()
