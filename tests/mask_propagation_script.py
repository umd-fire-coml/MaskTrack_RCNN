if __name__ == "__main__" and __package__ is None:
    __package__ = "tests"

from project.mask_propagation import MaskPropagationModule
import numpy as np

tst = np.empty((3000, 2000), dtype=np.float32)

mp = MaskPropagationModule()
mp.infer_mask(tst, tst, None)
