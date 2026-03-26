import argparse
import os
import datetime
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
import numpy as np

from RLBase.OptionClassifier import (train_val_split, 
                                     OptionRolloutDataset, 
                                     OptionClassifier, 
                                     plot_confusion_matrices, 
                                     plot_label_histogram,
                                     plot_option_lens_boxplot,
                                     )


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to the option rollout directory (e.g. Runs/Train/MiniGrid-.../VQOptionCritic/seed[123123])")
    
    # Name tag
    parser.add_argument("--name_tag", type=str, default="base",
                        help="Saved folder name tag (appended to 'Classifier/') for this experiment")
    
    # Network type
    parser.add_argument("--network", type=str, default="MiniGrid/Classifier/cnn",
                        help="Which network to use")
    
    # Feature type
    parser.add_argument("--feature", type=str, default="delta_last",
                        help="Which feature type to use")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu / cuda)")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Batch size
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    
    # Representation dimension
    parser.add_argument("--repr_dim", type=int, default=64,
                        help="Dimension of the representation layer")
    # Number of epochs
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")

    # Weighted loss
    parser.add_argument("--weighted_loss", action="store_true",
                        help="Use inverse-frequency class weights in CrossEntropyLoss")

    # Sampler type
    parser.add_argument("--sampler", type=str, default="random", choices=["random", "weighted"],
                        help="Sampler for the train loader: 'random' (shuffle) or 'weighted' (WeightedRandomSampler)")

    # KL distance-matrix regularizer weight
    parser.add_argument("--kl_weight", type=float, default=0.01,
                        help="Weight for the KL distance-matrix regularizer (0 = disabled)")

    return parser.parse_args()


def main():
    args = parse()

    train_path = Path(args.exp_dir)
    classifier_path = Path(str(train_path).replace("Runs/Train/", "Runs/Classifier/", 1))
    classifier_path = os.path.join(classifier_path, args.name_tag)
    os.makedirs(classifier_path, exist_ok=True)


    print("=" * 60)
    print(f"OptionClassifier training  [network={args.network}, feature={args.feature}]")
    print("=" * 60)
    print()

    torch.manual_seed(args.seed)

    # ---- dataset ----------------------------------------------------------------
    # Placeholder: config.DATASET_DICT will be wired up when Dataset files are ready
    dataset = OptionRolloutDataset(data_path=os.path.join(train_path, "option_rollouts_run1.pkl"),
                                    feature_type=args.feature, 
                                    encoder=None,
                                    gamma=0.99,
                                    num_actions=7,
                                    )

    print(f"  Records:      {len(dataset)}")
    print(f"  Num classes:  {dataset.num_classes}")
    print(f"  Feature dict:  {dataset.feature_dict}")
    print(f"  Class map:    {dataset.class_to_id}")
    class_names = [str(k) for k in sorted(dataset.class_to_id, key=dataset.class_to_id.get)]
    
    option_lens = dataset.option_lens()
    plot_option_lens_boxplot(option_lens, os.path.join(classifier_path, "option_lens_boxplot.png"))
    
    label_counts = dataset.label_counts()
    plot_label_histogram(label_counts, os.path.join(classifier_path, "label_histogram.png"))
    
    train_ds, val_ds = train_val_split(
        dataset,
        train_fraction=0.8,
        seed=args.seed,
    )
    print(f"  Train / Val:  {len(train_ds)} / {len(val_ds)}")
    print()

    if args.sampler == "weighted":
        sample_weights = torch.tensor(
            [1.0 / label_counts[dataset.class_to_id[dataset.labels[i].item()]] for i in train_ds.indices],
            dtype=torch.float,
        )
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False)
    
    # ---- classifier -------------------------------------------------------------
    if args.weighted_loss:
        counts = torch.tensor([label_counts[dataset.class_to_id[i]] for i in range(dataset.num_classes)], dtype=torch.float)
        class_weights = 1.0 / counts
        class_weights /= class_weights.sum()
    else:
        class_weights = None

    classifier_obj = OptionClassifier(
        feature_dict=dataset.feature_dict,
        num_classes=dataset.num_classes,
        network_type=args.network,
        repr_dim=args.repr_dim,
        class_weights=class_weights,
        kl_weight=args.kl_weight,
    )
    print(f"Classifier network:\n{classifier_obj.representation}\n")

    # ---- training loop ----------------------------------------------------------    
    history      = []
    best_val_acc = -1.0
    t_start      = time.time()
    print_freq   = 1

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = classifier_obj.train_epoch(train_loader)
        val_metrics   = classifier_obj.evaluate(val_loader)
        
        
        row = {
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "train_acc":  train_metrics["accuracy"],
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["accuracy"],
        }
        history.append(row)

        if print_freq > 0 and (epoch % print_freq == 0 or epoch == 1):
            elapsed = time.time() - t_start
            print(
                f"[{epoch:4d}/{args.num_epochs}]  "
                f"train loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.3f}  |  "
                f"val loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.3f}  "
                f"({elapsed:.1f}s)"
            )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            classifier_obj.save(os.path.join(classifier_path, "best_classifier.t"))
            train_conf_mat = classifier_obj.confusion_matrix(train_loader)
            val_conf_mat   = classifier_obj.confusion_matrix(val_loader)
        
            np.save(os.path.join(classifier_path, "confusion_matrix_best_train.npy"), train_conf_mat.numpy())
            np.save(os.path.join(classifier_path, "confusion_matrix_best_val.npy"),   val_conf_mat.numpy())
            plot_confusion_matrices(
                train_conf_mat, val_conf_mat,
                class_names=class_names,
                save_path=os.path.join(classifier_path, "confusion_matrices_best.png"),
            )

    # ---- confusion matrices (final epoch) ---------------------------------------
    train_conf_mat = classifier_obj.confusion_matrix(train_loader)
    val_conf_mat   = classifier_obj.confusion_matrix(val_loader)
    np.save(os.path.join(classifier_path, "confusion_matrix_final_train.npy"), train_conf_mat.numpy())
    np.save(os.path.join(classifier_path, "confusion_matrix_final_val.npy"),   val_conf_mat.numpy())
    plot_confusion_matrices(
        train_conf_mat, val_conf_mat,
        class_names=class_names,
        save_path=os.path.join(classifier_path, "confusion_matrices_final.png"),
    )
    

    # ---- save final model & history ---------------------------------------------
    classifier_obj.save(os.path.join(classifier_path, "final_classifier.t"))
    print(f"\nModels saved to {classifier_path}")

    with open(os.path.join(classifier_path, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}  "
          f"({time.time() - t_start:.1f}s total)")


if __name__ == "__main__":
    main()
