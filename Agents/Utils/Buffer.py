import random

class BasicBuffer:
    def __init__(self, capacity):
        self.buffer = []  # Empty buffer
        self.capacity = capacity
    
    @property
    def size(self):
        return len(self.buffer)  # Current buffer size
    
    def add_single_item(self, item):
        self.buffer.append(item)
        if self.size > self.capacity:
            self.buffer.pop(0)  # Remove oldest if over capacity
    
    def get_random_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)  # Random items
    
    def reset(self):
        self.buffer = []  # Clear the buffer

    def get_all(self):
        return self.buffer  # All items

    def remove_oldest(self):
        return self.buffer.pop(0)  # Remove oldest item
    
    def get_oldest(self):
        return self.buffer[0]  # Peek oldest item
    
    def is_full(self):
        return self.size == self.capacity  # Check capacity