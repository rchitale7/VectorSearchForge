from threading import Lock
from typing import List, TypeVar, Generic
from collections import deque

T = TypeVar('T')

class ThreadSafeRoundRobinIterator(Generic[T]):
    def __init__(self, items: List[T]):
        if not items:
            raise ValueError("Items list cannot be empty")
        # Create a copy of the input list to prevent external modifications
        self._items = deque(items)
        self._lock = Lock()
        self.current_pointer = 0

    def get_next(self) -> T:
        """
        Returns the next item in round-robin fashion in a thread-safe manner
        """
        with self._lock:
            # Get the first item
            item = self._items[self.current_pointer]
            # Move the pointer to the next item
            self.current_pointer = (self.current_pointer + 1) % len(self._items)
            return item

    def add_item(self, item: T) -> None:
        """
        Adds a new item to the collection in a thread-safe manner
        """
        with self._lock:
            self._items.append(item)

    def remove_item(self, item: T) -> None:
        """
        Removes an item from the collection in a thread-safe manner
        """
        with self._lock:
            try:
                self._items.remove(item)
                if not self._items:
                    raise ValueError("Cannot remove last item, iterator would be empty")
            except ValueError as e:
                raise ValueError(f"Item not found in the collection: {item}") from e
