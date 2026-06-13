import whisper
import json
import os
import glob

def create_annotations(segments_dir="./data/segments", output_file="./data/ground_truth.json"):
    model = whisper.load_model("medium")
    annotations = []

    segment_files = sorted(glob.glob(f"{segments_dir}/*.wav"))

    for seg_file in segment_files:
        print(f"\n{'='*50}")
        print(f"Fichier: {seg_file}")

        # Transcription automatique comme point de départ
        result = model.transcribe(seg_file, language="en")
        suggestion = result["text"].strip()
        print(f"Suggestion Whisper: {suggestion}")

        # Correction manuelle
        user_input = input("Transcription correcte (Enter = garder suggestion): ").strip()
        final_transcription = user_input if user_input else suggestion

        annotations.append({
            "file": seg_file,
            "reference": final_transcription
        })

        # Sauvegarde progressive
        with open(output_file, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"✅ Sauvegardé ({len(annotations)} annotations)")

    print(f"\n✅ Ground truth créé: {output_file}")
    return annotations

if __name__ == "__main__":
    create_annotations()
