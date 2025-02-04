from Evaluate.MultiExpAnalyzer import MultiExpAnalyzer

if __name__ == "__main__":
    # Define a dictionary where keys are experiment names and values are experiment directories.
    experiments = {
        "Best": "Runs/QLearningAgent(HyperParameters({'alpha': 0.284, 'gamma': 0.99, 'epsilon': 0.078}))_seed [2500]_20250204_163353",
        "Not Best": "Runs/QLearningAgent(HyperParameters({'alpha': 0.5, 'gamma': 0.99, 'epsilon': 0.1}))_seed [2500]_20250204_163410",
        # Add more experiments as needed.
    }
    
    analyzer = MultiExpAnalyzer(experiments)
    analyzer.load_metrics()  # Loads metrics.pkl from each experiment directory.
    analyzer.plot_combined(save_path="Runs/combined_rewards.png")