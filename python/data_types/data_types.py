from enum import Enum


class IndexTypes(Enum):
    CPU = 'cpu'
    GPU = 'gpu'

    @staticmethod
    def from_str(labelstr: str):
        if labelstr in 'cpu':
            return IndexTypes.CPU
        elif labelstr in 'gpu':
            return IndexTypes.GPU
        else:
            raise NotImplementedError
