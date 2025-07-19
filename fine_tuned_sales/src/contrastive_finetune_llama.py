# src/contrastive_finetune_llama.py

import os
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, losses, InputExample

# Resolve paths relative to this file
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "sales_transcripts.csv")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "models", "llama_contrastive")

def load_sales_transcripts(path=DATA_PATH):
    """
    Load transcripts and conversion labels from CSV.
    """
    df = pd.read_csv(path)
    transcripts = df["transcript"].tolist()
    labels      = df["label"].tolist()
    return transcripts, labels

def create_contrastive_training_examples(transcripts, labels):
    """
    Build InputExample pairs:
      - label=1.0 for (won, won)
      - label=0.0 for (won, lost)
    """
    examples   = []
    won_idxs   = [i for i, l in enumerate(labels) if l == "won"]
    lost_idxs  = [i for i, l in enumerate(labels) if l == "lost"]

    # Positive pairs: adjacent won calls
    for idx in range(len(won_idxs) - 1):
        text_a = transcripts[won_idxs[idx]]
        text_b = transcripts[won_idxs[idx + 1]]
        examples.append(InputExample(texts=[text_a, text_b], label=1.0))

    # Negative pairs: pair each won with a lost
    for idx in range(min(len(won_idxs), len(lost_idxs))):
        text_a = transcripts[won_idxs[idx]]
        text_b = transcripts[lost_idxs[idx]]
        examples.append(InputExample(texts=[text_a, text_b], label=0.0))

    return examples

def main():
    # 1) Load data & build contrastive examples
    transcripts, labels = load_sales_transcripts()
    contrastive_examples = create_contrastive_training_examples(transcripts, labels)
    print(f"→ Generated {len(contrastive_examples)} contrastive training examples")

    # 2) Build the Llama-backed SentenceTransformer
    transformer_module = models.Transformer("meta-llama/Llama-2-7b-hf")
    pooling_module     = models.Pooling(transformer_module.get_word_embedding_dimension())
    llama_encoder      = SentenceTransformer(modules=[transformer_module, pooling_module])

    # 3) Create DataLoader & define loss
    dataloader     = DataLoader(contrastive_examples, batch_size=8, shuffle=True)
    contrastive_loss = losses.ContrastiveLoss(model=llama_encoder)

    # 4) Fine-tune the encoder
    llama_encoder.fit(
        train_objectives=[(dataloader, contrastive_loss)],
        epochs=3,
        output_path=OUTPUT_DIR,
        show_progress_bar=True
    )

    print(f"✅ Contrastive fine‑tuning complete. Model saved to:\n   {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
