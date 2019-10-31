class Experience(object):
    def __init__(self, max_memory_size) -> None:
        self.memory = []
        self.memory_index = 0
        self.max_memory_size = max_memory_size

    def add_experience(self, state, action, reward, next_state, done):
        if len(self.memory) < self.max_memory_size:
            self.memory.append(None)
        self.memory[self.memory_index] = (state, action, reward, next_state, done)
        self.memory_index = (self.memory_index + 1) % self.max_memory_size

    def empty_memory(self):
        self.memory = []
        self.memory_index = 0

    def is_memory_full(self):
        return len(self.memory) == self.max_memory_size
