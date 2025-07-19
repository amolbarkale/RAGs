# src/contrastive_finetune_bert.py

import os
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, losses, InputExample

# Resolve paths relative to this file
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "sales_transcripts.csv")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "models", "bert_contrastive")

def load_sales_transcripts(path=DATA_PATH):
    """
    Load call transcripts and their 'won'/'lost' labels.
    """
    df = pd.read_csv(path)
    return df["transcript"].tolist(), df["label"].tolist()

def build_contrastive_examples(transcripts, labels):
    """
    Create supervised contrastive pairs:
      - label=1.0 for (won, won)
      - label=0.0 for (won, lost)
    """
    examples  = []
    won_idxs  = [i for i, l in enumerate(labels) if l == "won"]
    lost_idxs = [i for i, l in enumerate(labels) if l == "lost"]

    # positive pairs: consecutive won calls
    for idx in range(len(won_idxs) - 1):
        a = transcripts[won_idxs[idx]]
        b = transcripts[won_idxs[idx + 1]]
        examples.append(InputExample(texts=[a, b], label=1.0))

    # negative pairs: match each won with a lost
    for idx in range(min(len(won_idxs), len(lost_idxs))):
        a = transcripts[won_idxs[idx]]
        b = transcripts[lost_idxs[idx]]
        examples.append(InputExample(texts=[a, b], label=0.0))

    return examples

def main():
    # 1) Load data & build examples
    transcripts, labels = load_sales_transcripts()
    contrastive_examples = build_contrastive_examples(transcripts, labels)
    print(f"→ Created {len(contrastive_examples)} contrastive examples")

    # 2) Build BERT-backed SentenceTransformer
    bert_transformer = models.Transformer("bert-base-uncased")
    pooling_layer    = models.Pooling(bert_transformer.get_word_embedding_dimension())
    encoder          = SentenceTransformer(modules=[bert_transformer, pooling_layer])

    # 3) Wrap in DataLoader + define loss
    train_loader = DataLoader(contrastive_examples, batch_size=8, shuffle=True)
    contrastive_loss = losses.ContrastiveLoss(model=encoder)

    # 4) Fine‑tune & save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    encoder.fit(
        train_objectives=[(train_loader, contrastive_loss)],
        epochs=3,
        output_path=OUTPUT_DIR,
        show_progress_bar=True
    )
    print(f"✅ Contrastive fine‑tuning with BERT complete. Model saved to:\n   {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
