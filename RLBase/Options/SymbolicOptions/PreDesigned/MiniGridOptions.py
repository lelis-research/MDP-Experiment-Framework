from ...Base import BaseOption
from ....registry import register_option

@register_option
class test_option(BaseOption):
    def __init__(self):
        self.counter = 0
        self.option_id = "test"
        self.hp = None
        self.device = None

    def select_action(self, obsersvation):
        self.counter += 1
        return 2

    def is_terminated(self, observation):
        if self.counter >= 10:
            self.counter = 0
            return True
        return False
    
    def reset(self):
        self.counter = 0
