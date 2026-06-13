import whisper
import json
import numpy as np
from jiwer import wer
import os

def evaluate(ground_truth_file="./data/ground_truth.json", output_file="./data/results.json"):
    model = whisper.load_model("small")

    with open(ground_truth_file) as f:
        annotations = json.load(f)

    results = []
    prompt = "Radio communication, callsigns, military, aviation, Alpha Bravo Charlie."

    for item in annotations:
        ref = item["reference"].lower()
        seg_file = item["file"]

        # Chemin vers la version pre-processée
        processed_file = seg_file.replace("segments", "processed")

        if not os.path.exists(processed_file):
            print(f"⚠️ Pas de version processed pour {seg_file}, skip")
            continue

        # Baseline — audio brut sans prompt
        pred_baseline = model.transcribe(seg_file, language="en")["text"].lower()

        # Amélioré — audio processé + prompt
        pred_improved = model.transcribe(
            processed_file,
            language="en",
            initial_prompt=prompt
        )["text"].lower()

        wer_b = wer(ref, pred_baseline)
        wer_i = wer(ref, pred_improved)

        results.append({
            "file": seg_file,
            "reference": ref,
            "baseline": pred_baseline,
            "improved": pred_improved,
            "wer_baseline": wer_b,
            "wer_improved": wer_i,
        })

        print(f"\n📄 {os.path.basename(seg_file)}")
        print(f"  Ref:      {ref}")
        print(f"  Baseline: {pred_baseline} (WER: {wer_b:.1%})")
        print(f"  Improved: {pred_improved} (WER: {wer_i:.1%})")

    # Moyennes
    avg_baseline = np.mean([r["wer_baseline"] for r in results])
    avg_improved = np.mean([r["wer_improved"] for r in results])
    gain = avg_baseline - avg_improved

    summary = {
        "wer_baseline": avg_baseline,
        "wer_improved": avg_improved,
        "gain": gain,
        "details": results
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ WER Baseline:  {avg_baseline:.1%}")
    print(f"✅ WER Amélioré:  {avg_improved:.1%}")
    print(f"🚀 Gain:          {gain:.1%}")

if __name__ == "__main__":
    evaluate()
