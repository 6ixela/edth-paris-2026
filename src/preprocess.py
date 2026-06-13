import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from scipy.signal import butter, filtfilt
import os
import glob

def bandpass_filter(audio, sr, lowcut=300, highcut=3400):
    """Filtre passe-bande typique radio militaire"""
    nyq = sr / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, audio)

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val * 0.9
    return audio

def ptt_preprocess(input_path, output_path):
    """Pipeline complet de pre-processing"""
    audio, sr = librosa.load(input_path, sr=16000)

    # 1. Réduction de bruit
    audio = nr.reduce_noise(y=audio, sr=sr, stationary=False)

    # 2. Filtre passe-bande radio
    audio = bandpass_filter(audio, sr)

    # 3. Normalisation
    audio = normalize_audio(audio)

    sf.write(output_path, audio, sr)
    return output_path

if __name__ == "__main__":
    input_dir = "./data/segments"
    output_dir = "./data/processed"
    os.makedirs(output_dir, exist_ok=True)

    for input_file in sorted(glob.glob(f"{input_dir}/*.wav")):
        filename = os.path.basename(input_file)
        output_file = f"{output_dir}/{filename}"
        ptt_preprocess(input_file, output_file)
        print(f"✅ Processed: {filename}")
