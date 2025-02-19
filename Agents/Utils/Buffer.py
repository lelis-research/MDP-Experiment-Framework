import random

class BasicBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
    
    @property
    def size(self):
        return len(self.buffer)
    
    def add_single_item(self, item):
        self.buffer.append(item)
        if self.size > self.capacity:
            self.buffer.pop(0)
    
    def get_random_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def reset(self):
        self.buffer = []

    def get_all(self):
        return self.buffer

    def remove_oldest(self):
        return self.buffer.pop(0)
    
    def get_oldest(self):
        return self.buffer[0]
    
    def is_full(self):
        return self.size == self.capacity
    