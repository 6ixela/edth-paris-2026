from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import json
import torch
import soundfile as sf
import numpy as np
import librosa
import os

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad input_features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Remplace padding token par -100 (ignoré dans la loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


def load_dataset_from_annotations(ground_truth_file="./data/ground_truth.json"):
    with open(ground_truth_file) as f:
        annotations = json.load(f)

    for item in annotations[:3]:
        path = item["file"]
        print(f"{'✅' if os.path.exists(path) else '❌'} {path}")

    data = {
        "audio_path": [item["file"] for item in annotations],
        "sentence": [item["reference"] for item in annotations]
    }
    return Dataset.from_dict(data)


def finetune():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    dataset = load_dataset_from_annotations()

    def prepare_batch(batch):
        audio_array, sr = sf.read(batch["audio_path"])

        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        if sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

        inputs = processor(
            audio_array.astype(np.float32),
            sampling_rate=16000,
            return_tensors="pt"
        )
        batch["input_features"] = inputs.input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    dataset = dataset.map(prepare_batch, remove_columns=["audio_path", "sentence"])

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/whisper-radio-finetuned",
        per_device_train_batch_size=4,
        num_train_epochs=10,
        save_steps=100,
        logging_steps=10,
        learning_rate=1e-5,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,  # ← clé du fix
    )

    trainer.train()
    model.save_pretrained("./models/whisper-radio-finetuned")
    processor.save_pretrained("./models/whisper-radio-finetuned")
    print("✅ Modèle sauvegardé")


if __name__ == "__main__":
    finetune()
