import random as r


class RandomArray:
    def __init__(self):
        self.array = []
        self.max = 0

    def insert(self, item):
        # put in array
        self.array.append(item)
        if len(self.array) > self.max:
            self.max = len(self.array)

    def delete_max(self):
        return self.array.pop(r.randint(0, len(self.array) - 1))

    def size(self):
        return len(self.array)

    def prune(self, limit):
        num_pruned = 0
        i = 0
        while i < len(self.array):
            if self.array[i].lower_bound > limit:
                self.array.pop(i)
                i -= 1
                num_pruned += 1
            i += 1
        return num_pruned
