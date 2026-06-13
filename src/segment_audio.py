import librosa
import soundfile as sf
import os

def split_into_segments(input_file, output_dir="./data/segments", segment_duration=20):
    os.makedirs(output_dir, exist_ok=True)
    audio, sr = librosa.load(input_file, sr=16000)
    total_duration = len(audio) / sr
    segments = []
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    for i, start in enumerate(range(0, int(total_duration), segment_duration)):
        end = min(start + segment_duration, int(total_duration))
        segment = audio[start*sr : end*sr]

        if len(segment) < sr * 2:
            continue

        output_path = os.path.join(output_dir, f"{base_name}_segment_{i:03d}.wav")
        sf.write(output_path, segment, sr)
        segments.append(output_path)
        print(f"✅ Segment {i}: {start}s - {end}s -> {output_path}")

    return segments

if __name__ == "__main__":
    import glob
    raw_files = sorted(glob.glob("./data/raw/*.wav"))
    for f in raw_files:
        print(f"Traitement: {f}")
        split_into_segments(f)
