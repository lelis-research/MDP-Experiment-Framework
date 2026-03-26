import argparse
import os
import json
from pathlib import Path
import torch
import numpy as np

from RLBase.OptionClassifier import (OptionRolloutDataset, OptionClassifier)
from RLBase.OptionEmbedding import OptionEmbedding, plot_distance_matrix, plot_embeddings_2d


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to the training run directory (e.g. Runs/Train/MiniGrid-.../VQOptionCritic/seed[123123])")

    # Which classifier checkpoint to load
    parser.add_argument("--classifier_tag", type=str, default="base",
                        help="Name tag of the classifier (subdirectory under Runs/Classifier/...)")
    parser.add_argument("--classifier_ckpt", type=str, default="best_classifier.t",
                        help="Classifier checkpoint file name (e.g. best_classifier.t / final_classifier.t)")

    # Name tag for this embedding run
    parser.add_argument("--name_tag", type=str, default="base",
                        help="Saved folder name tag (appended to 'Embedding/') for this experiment")

    # Dataset args (must match classifier training)
    parser.add_argument("--feature", type=str, default="delta_last",
                        help="Feature type used when the classifier was trained")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for feature extraction")
    parser.add_argument("--num_actions", type=int, default=7,
                        help="Number of actions in the environment")

    # Embedding args
    parser.add_argument("--mode", type=str, default="repr", choices=["repr", "repr_pca"],
                        help="Embedding mode: 'repr' (full repr dim) or 'repr_pca' (PCA projection)")
    parser.add_argument("--pca_dim", type=int, default=4,
                        help="Target PCA dimension (only used when mode=repr_pca)")
    parser.add_argument("--no_normalize_rollouts", action="store_true",
                        help="Disable L2 normalisation of per-sample representations before averaging")
    parser.add_argument("--no_normalize_class_means", action="store_true",
                        help="Disable L2 normalisation of class mean embeddings")

    # Misc
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for representation extraction")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu / cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    args = parse()

    torch.manual_seed(args.seed)

    train_path = Path(args.exp_dir)
    classifier_path = Path(str(train_path).replace("Runs/Train/", "Runs/Classifier/", 1)) / args.classifier_tag
    embedding_path  = Path(str(train_path).replace("Runs/Train/", "Runs/Embedding/", 1)) / args.classifier_tag / args.name_tag
    
    os.makedirs(embedding_path, exist_ok=True)

    print("=" * 60)
    print(f"OptionEmbedding  [mode={args.mode}, feature={args.feature}]")
    print("=" * 60)

    # ---- dataset ----------------------------------------------------------------
    dataset = OptionRolloutDataset(
        data_path=os.path.join(train_path, "option_rollouts_run1.pkl"),
        feature_type=args.feature,
        encoder=None,
        gamma=args.gamma,
        num_actions=args.num_actions,
    )

    print(f"  Records:     {len(dataset)}")
    print(f"  Num classes: {dataset.num_classes}")
    print(f"  Feature dict: {dataset.feature_dict}")
    print(f"  Class map:   {dataset.class_to_id}")
    print()

    # ---- load classifier --------------------------------------------------------
    ckpt_file = os.path.join(classifier_path, args.classifier_ckpt)
    print(f"Loading classifier from: {ckpt_file}")
    classifier = OptionClassifier.load(ckpt_file)
    print(f"Classifier repr dim: {classifier.repr_dim}")
    print()

    # ---- build embeddings -------------------------------------------------------
    embedding = OptionEmbedding(
        classifier=classifier,
        dataset=dataset,
        mode=args.mode,
        pca_dim=args.pca_dim,
        normalize_rollouts=not args.no_normalize_rollouts,
        normalize_class_means=not args.no_normalize_class_means,
        device=args.device,
        batch_size=args.batch_size,
    )

    embeddings = embedding.get_embeddings()  # (num_options, emb_dim)
    print(f"\nEmbedding table shape: {tuple(embeddings.shape)}")
    print(f"Per-class sample counts: {embedding.counts.tolist()}")
    if embedding.explained_variance_ratio is not None:
        print(f"PCA explained variance ratio: {embedding.explained_variance_ratio.tolist()}")
        print(f"  sum = {embedding.explained_variance_ratio.sum().item():.4f}")

    # ---- save -------------------------------------------------------------------
    emb_file = os.path.join(embedding_path, "embeddings.t")
    embedding.save(emb_file)
    print(f"\nEmbedding saved to: {emb_file}")

    np.save(os.path.join(embedding_path, "embeddings.npy"), embeddings.numpy())

    # ---- visualisations ---------------------------------------------------------
    plot_distance_matrix(
        embeddings,
        save_path=os.path.join(embedding_path, "dist_matrix_cosine.png"),
        metric="cosine",
    )
    plot_distance_matrix(
        embeddings,
        save_path=os.path.join(embedding_path, "dist_matrix_l2.png"),
        metric="l2",
    )
    plot_embeddings_2d(
        embeddings,
        save_path=os.path.join(embedding_path, "embeddings_2d.png"),
        title=f"Option Embeddings 2D PCA [{args.mode}]",
    )
    print("Plots saved.")

    meta = {
        "exp_dir":                str(args.exp_dir),
        "classifier_tag":         args.classifier_tag,
        "classifier_ckpt":        args.classifier_ckpt,
        "name_tag":               args.name_tag,
        "feature":                args.feature,
        "mode":                   args.mode,
        "pca_dim":                args.pca_dim,
        "normalize_rollouts":     not args.no_normalize_rollouts,
        "normalize_class_means":  not args.no_normalize_class_means,
        "embedding_shape":        list(embeddings.shape),
        "counts":                 embedding.counts.tolist(),
        "explained_variance_ratio": (
            embedding.explained_variance_ratio.tolist()
            if embedding.explained_variance_ratio is not None else None
        ),
        "embedding":              embeddings.numpy().tolist(),
    }
    with open(os.path.join(embedding_path, "embedding_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Metadata saved to: {os.path.join(embedding_path, 'embedding_meta.json')}")
    print("\nDone.")


if __name__ == "__main__":
    main()