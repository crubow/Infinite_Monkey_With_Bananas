class PriorityQueueHeap:
    def __init__(self):
        self.queue = []
        self.max = 0

    def insert(self, item):
        # put on end of array
        self.queue.append(item)

        if len(self.queue) > self.max:
            self.max = len(self.queue)

        # move the new node up the tree if needs be
        current = len(self.queue) - 1
        next_node = (current - 1) // 2
        while next_node >= 0:
            if self.queue[next_node].get_depth() >= item.get_depth():
                break
            # swapping nodes
            self.queue[next_node], self.queue[current] = self.queue[current], self.queue[next_node]
            current = next_node
            next_node = (next_node - 1) // 2

    def delete_max(self):
        if len(self.queue) == 0:
            return None

        # swap min node with bottom node for easy heapifying
        self.queue[0], self.queue[len(self.queue) - 1] = self.queue[len(self.queue) - 1], self.queue[0]

        minimum = self.queue.pop(len(self.queue) - 1)

        # heapify
        if len(self.queue) != 0:
            self.heapify(0)

        return minimum

    def heapify(self, pos):
        left = pos*2 + 1
        # make sure such a node exists
        if left >= len(self.queue):
            left = None

        right = pos*2 + 2
        # make sure such a node exists
        if right >= len(self.queue):
            right = None

        # this means we are at a leaf (no children)
        if left is None and right is None:
            return

        # only has one leaf
        if left is not None and right is None:
            if self.queue[left].get_depth() > self.queue[pos].get_depth():
                self.queue[left], self.queue[pos] = self.queue[pos], self.queue[left]
            # no need for any more recursion as next node is a leaf
            return

        # only has one leaf
        if right is not None and left is None:
            if self.queue[right].get_depth() > self.queue[pos].get_depth():
                self.queue[right], self.queue[pos] = self.queue[pos], self.queue[right]
            # no need for any more recursion as next node is a leaf
            return

        # see if either children are less than root
        if self.queue[pos].get_depth() < self.queue[right].get_depth() or \
                self.queue[pos].get_depth() < self.queue[left].get_depth():
            # see which child is lesser
            if self.queue[right].get_depth() > self.queue[left].get_depth():
                self.queue[right], self.queue[pos] = self.queue[pos], self.queue[right]
                self.heapify(right)
            else:
                self.queue[left], self.queue[pos] = self.queue[pos], self.queue[left]
                self.heapify(left)

    def size(self):
        return len(self.queue)

    def prune(self, limit):
        num_pruned = 0
        i = 0
        while i < len(self.queue):
            if self.queue[i].lower_bound > limit:
                self.queue.pop(i)
                i -= 1
                num_pruned += 1
            i += 1
        self.heapify(0)
        return num_pruned
