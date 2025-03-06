class BaseOption:
    def __init__(self, feature_extractor, num_options=1):
        '''
        For implmentation simplicity each Option consists of multiple options
        showing with num_options because they might share the same basis and
        this way it will save us memory and compute
        '''
        self.feature_extractor = feature_extractor
        self.policy = None
        self.n = num_options
        
    def select_action(self, observation):
        pass
    
    def is_terminated(self, observation):
        pass