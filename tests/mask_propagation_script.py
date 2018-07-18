import tests.context
from project.mask_propagation import MaskPropagationModule
import numpy as np

tst = np.empty((3000, 2000), dtype=np.float32)

mp = MaskPropagationModule()
mp.infer_mask(tst, tst, None)
