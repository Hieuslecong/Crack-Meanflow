import numpy as np, pytest
from PIL import Image
from crackmeanflow.common.data import _mask_binary_array

def test_auto_binary_safe_accepts_0_1():
    a=np.array([[0,1]],dtype=np.uint8)
    assert _mask_binary_array(Image.fromarray(a),'auto_binary_safe').tolist()==[[0.0,1.0]]

def test_auto_binary_safe_accepts_0_255():
    a=np.array([[0,255]],dtype=np.uint8)
    assert _mask_binary_array(Image.fromarray(a),'auto_binary_safe').tolist()==[[0.0,1.0]]

def test_auto_binary_safe_rejects_ambiguous_low_range():
    a=np.array([[0,2]],dtype=np.uint8)
    with pytest.raises(RuntimeError,match='ambiguous low-range'):
        _mask_binary_array(Image.fromarray(a),'auto_binary_safe')
