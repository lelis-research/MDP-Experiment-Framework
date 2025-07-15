from ...loaders import load_policy, load_feature_extractor

class BaseOption:
    def __init__(self, feature_extractor, policy):
        '''
        For implmentation simplicity each Option consists of multiple options
        showing with num_options because they might share the same basis and
        this way it will save us memory and compute
        '''
        self.feature_extractor = feature_extractor
        self.policy = policy
        
    def select_action(self, observation):
        pass
    
    def is_terminated(self, observation):
        pass
    
    def save(self, file_path=None):
        policy_checkpoint = self.policy.save()
        feature_extractor_checkpoint = self.feature_extractor.save()
        checkpoint = {
            'feature_extractor':feature_extractor_checkpoint,
            'policy': policy_checkpoint,
            'option_class': self.__class__.__name__,
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_options.t")
        return checkpoint


    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        """
        Load the Q-table and hyper-parameters.
        
        Args:
            file_path (str): File path from which to load the checkpoint.
        """
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        policy = load_policy("", checkpoint=checkpoint['policy'])
        feature_extractor = load_feature_extractor("", checkpoint=checkpoint['feature_extractor'])
        instance = cls(feature_extractor, policy)

        return instance