from pathlib import Path

import torch

from tech_support_ai.classifier import train_classifier
from tech_support_ai.datasets import ISSUE_LABELS, SEED_TRAINING_EXAMPLES


def main() -> None:
    trained = train_classifier(SEED_TRAINING_EXAMPLES, ISSUE_LABELS)
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "classifier.pt"
    torch.save(
        {
            "state_dict": trained.model.state_dict(),
            "token_to_idx": trained.vectorizer.token_to_idx,
            "labels": list(trained.labels),
        },
        ckpt_path,
    )
    print(f"Saved classifier checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
