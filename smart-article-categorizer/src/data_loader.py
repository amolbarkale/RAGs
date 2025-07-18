# src/data_loader.py

from datasets import load_dataset
from typing import List, Tuple
from sklearn.model_selection import train_test_split

def load_hf_datasets(
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load and split three HuggingFace datasets into training/testing splits.

    Returns:
        training_texts:  List of article strings for training
        testing_texts:   List of article strings for testing
        training_labels: List of labels for training
        testing_labels:  List of labels for testing
    """

    # 1) Education
    edu_dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="CC-MAIN-2024-10",
        split="train",
        streaming=False
    )
    education_texts   = [record["text"] for record in edu_dataset]
    education_labels  = ["Education"] * len(education_texts)

    # 2) Technology
    tech_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    technology_texts  = [record["question"] for record in tech_dataset]
    technology_labels = [record["label"]    for record in tech_dataset]

    # 3) Sports
    sports_dataset = load_dataset(
        "BobbleAI/Bobble-Hinglish-Sports-Dataset_BHSD",
        split="train"
    )
    sports_texts   = [record["Description"] for record in sports_dataset]
    sports_labels  = [record["Labels"]      for record in sports_dataset]

    # Combine all sources
    all_texts = education_texts + technology_texts + sports_texts
    all_labels = education_labels + technology_labels + sports_labels

    # Stratified train/test split
    training_texts, testing_texts, training_labels, testing_labels = train_test_split(
        all_texts,
        all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=random_state
    )

    return training_texts, testing_texts, training_labels, testing_labels
