import argparse
import os
import json
from pathlib import Path
import torch
import numpy as np

from RLBase.OptionClassifier import (OptionRolloutDataset, OptionClassifier)
from RLBase.OptionEmbedding import OptionEmbedding, plot_distance_matrix, plot_embeddings_2d
from RLBase import load_policy, load_agent


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to the training run directory (e.g. Runs/Train/MiniGrid-.../VQOptionCritic/seed[123123])")
    # Which model
    parser.add_argument("--model_type", type=str, choices=["encoder", "classifier", "feature"], default="encoder",
                        help="Using the classifier representation or agent encoder")

    # Which classifier checkpoint to load
    parser.add_argument("--classifier_tag", type=str, default="base",
                        help="Name tag of the classifier (subdirectory under Runs/Classifier/...)")
    parser.add_argument("--classifier_ckpt", type=str, default="last_classifier.t",
                        help="Classifier checkpoint file name (e.g. best_classifier.t / final_classifier.t)")

    # Name tag for this embedding run
    parser.add_argument("--name_tag", type=str, default="",
                        help="Saved folder name tag (appended to 'Embedding/') for this experiment")

    # Dataset args (must match classifier training)
    parser.add_argument("--feature", type=str, default="delta_last", choices=["last", "delta_last", "sf", "delta_sf", "reverse_sf", "delta_reverse_sf", 
                                                                              "last_enc", "delta_last_enc", "sf_enc", "delta_sf_enc", "reverse_sf_enc", "delta_reverse_sf_enc"],
                        help="Feature type used when the classifier was trained")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for feature extraction")

    # Embedding args
    parser.add_argument("--mode", type=str, default="repr",
                        choices=["repr", "repr_pca", "repr_learned", "repr_pca_learned"],
                        help="Embedding mode: 'repr' | 'repr_pca' | 'repr_learned' | 'repr_pca_learned'")
    parser.add_argument("--pca_dim", type=int, default=4,
                        help="Target PCA dimension (used when mode contains 'pca')")
    parser.add_argument("--no_normalize_rollouts", action="store_true",
                        help="Disable L2 normalisation of per-sample representations before averaging")
    parser.add_argument("--no_normalize_class_means", action="store_true",
                        help="Disable L2 normalisation of class mean embeddings")

    # Learned-mode args
    parser.add_argument("--emb_dim", type=int, default=4,
                        help="Embedding table dimension (repr_learned / repr_pca_learned)")
    parser.add_argument("--kl_weight", type=float, default=0.05,
                        help="Weight on the KL structure-matching term")
    parser.add_argument("--kl_metric", type=str, default="cosine", choices=["cosine", "l2"],
                        help="Distance metric for KL distance matrices")
    parser.add_argument("--kl_temperature", type=float, default=1.0,
                        help="Softmax temperature for distance-to-probability conversion")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Training epochs (repr_learned / repr_pca_learned)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (repr_learned / repr_pca_learned)")

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
    learned_suffix = (
        f"_emb{args.emb_dim}_kl{args.kl_weight}_{args.kl_metric}_t{args.kl_temperature}"
        if args.mode in ("repr_learned", "repr_pca_learned") else ""
    )
    mode_tag = f"{args.mode}{learned_suffix}"

    if args.model_type == "classifier":
        embedding_path = Path(str(train_path).replace("Runs/Train/", "Runs/Embedding/", 1)) / args.classifier_tag / f"{args.feature}_{mode_tag}_{args.name_tag}"
    elif args.model_type == "encoder":
        embedding_path = Path(str(train_path).replace("Runs/Train/", "Runs/Embedding/", 1)) / "AgentEncoder" / f"{args.feature}_{mode_tag}_{args.name_tag}"
    elif args.model_type == "feature":
        embedding_path = Path(str(train_path).replace("Runs/Train/", "Runs/Embedding/", 1)) / "AgentEncoder-Feature" / f"{args.feature}_{mode_tag}_{args.name_tag}"
        
    os.makedirs(embedding_path, exist_ok=True)

    print("=" * 60)
    print(f"OptionEmbedding  [mode={args.mode}, feature={args.feature}]")
    print("=" * 60)

    # ---- dataset ----------------------------------------------------------------
    ckpt_file = os.path.join(train_path, "Run1_Last_agent.t")
    agent = load_agent(ckpt_file)  # to ensure agent code is loaded (for custom envs, wrappers, etc.)
    encoder = agent.encoder.encoder
    dataset = OptionRolloutDataset(
        data_path=os.path.join(train_path, "option_rollouts_run1.pkl"),
        feature_type=args.feature,
        encoder=encoder,
        gamma=args.gamma,
    )

    print(f"  Records:     {len(dataset)}")
    print(f"  Num classes: {dataset.num_classes}")
    print(f"  Feature dict: {dataset.feature_dict}")
    print(f"  Class map:   {dataset.class_to_id}")
    print()

    # ---- load classifier --------------------------------------------------------
    if args.model_type == "classifier":
        ckpt_file = os.path.join(classifier_path, args.classifier_ckpt)
        print(f"Loading classifier from: {ckpt_file}")
        model = OptionClassifier.load(ckpt_file)
        print(f"Using classifier as model. Repr dim: {model.repr_dim}")
        print()
    elif args.model_type == "encoder":
        ckpt_file = os.path.join(train_path, "Run1_Last_agent.t")
        print(f"Loading agent from: {ckpt_file}")
        agent = load_agent(ckpt_file)  # to ensure agent code is loaded (for custom envs, wrappers, etc.)
        model = agent.encoder.encoder
        print(f"Using agent encoder as model. Repr dim: {agent.encoder.hp.enc_dim}")
        print()
    elif args.model_type == "feature":
        model = None
        print("Using raw features (no model). Repr dim = sum of feature dims.")
        print()
    
    # ---- build embeddings -------------------------------------------------------
    embedding = OptionEmbedding(
        model=model,
        dataset=dataset,
        mode=args.mode,
        pca_dim=args.pca_dim,
        normalize_rollouts=not args.no_normalize_rollouts,
        normalize_class_means=not args.no_normalize_class_means,
        device=args.device,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        kl_weight=args.kl_weight,
        kl_metric=args.kl_metric,
        kl_temperature=args.kl_temperature,
        num_epochs=args.num_epochs,
        lr=args.lr,
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
        "model_type":             args.model_type,
        "classifier_tag":         args.classifier_tag,
        "classifier_ckpt":        args.classifier_ckpt,
        "name_tag":               args.name_tag,
        "feature":                args.feature,
        "mode":                   args.mode,
        "pca_dim":                args.pca_dim,
        "normalize_rollouts":     not args.no_normalize_rollouts,
        "normalize_class_means":  not args.no_normalize_class_means,
        "emb_dim":                args.emb_dim,
        "kl_weight":              args.kl_weight,
        "kl_metric":              args.kl_metric,
        "kl_temperature":         args.kl_temperature,
        "num_epochs":             args.num_epochs,
        "lr":                     args.lr,
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