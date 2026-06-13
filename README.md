# edth-paris-2026

## Structure du dépôt

- `data/raw/` : fichiers audio sources
- `data/segments/` : segments générés à partir des fichiers raw
- `data/processed/` : segments pré-traités pour Whisper
- `data/ground_truth.json` : annotations de référence
- `data/results.json` : résultats d'évaluation

## Exécution

1. `python src/segment_audio.py`
2. `python src/preprocess.py`
3. `python src/create_truth.py`
4. `python src/evaluate.py`

## Notes

Le pipeline s'appuie sur des fichiers segmentés et traités dans des dossiers cohérents : `data/segments` et `data/processed`. Les scripts utilisent des chemins relatifs pour faciliter l'utilisation depuis la racine du dépôt.
