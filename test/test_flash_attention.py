"""Tests for CUDA Flash Attention implementation."""
import unittest
import os
os.environ['FLASH_ATTENTION'] = '1'

from tinygrad import Tensor, Device, dtypes
import numpy as np

def has_cuda():
  try:
    return 'CUDA' in Device.DEFAULT
  except:
    return False

@unittest.skipUnless(has_cuda(), 'CUDA required')
class TestFlashAttention(unittest.TestCase):
  
  def test_flash_attention_basic(self):
    """Test basic flash attention matches standard SDPA."""
    B, H, N, D = 2, 8, 128, 64
    q = Tensor.randn(B, H, N, D, device='CUDA').realize()
    k = Tensor.randn(B, H, N, D, device='CUDA').realize()
    v = Tensor.randn(B, H, N, D, device='CUDA').realize()
    
    # Flash attention result
    os.environ['FLASH_ATTENTION'] = '1'
    flash_out = q.scaled_dot_product_attention(k, v).numpy()
    
    # Standard SDPA result
    os.environ['FLASH_ATTENTION'] = '0'
    sdpa_out = q.scaled_dot_product_attention(k, v).numpy()
    
    np.testing.assert_allclose(flash_out, sdpa_out, rtol=1e-2, atol=1e-2)
  
  def test_flash_attention_causal(self):
    """Test causal flash attention."""
    B, H, N, D = 2, 8, 128, 64
    q = Tensor.randn(B, H, N, D, device='CUDA').realize()
    k = Tensor.randn(B, H, N, D, device='CUDA').realize()
    v = Tensor.randn(B, H, N, D, device='CUDA').realize()
    
    os.environ['FLASH_ATTENTION'] = '1'
    flash_out = q.scaled_dot_product_attention(k, v, is_causal=True).numpy()
    
    os.environ['FLASH_ATTENTION'] = '0'
    sdpa_out = q.scaled_dot_product_attention(k, v, is_causal=True).numpy()
    
    np.testing.assert_allclose(flash_out, sdpa_out, rtol=1e-2, atol=1e-2)
  
  def test_flash_attention_shapes(self):
    """Test various batch and sequence length combinations."""
    for B in [1, 4, 8]:
      for N in [64, 128, 256, 512]:
        with self.subTest(B=B, N=N):
          q = Tensor.randn(B, 16, N, 64, device='CUDA').realize()
          k = Tensor.randn(B, 16, N, 64, device='CUDA').realize()
          v = Tensor.randn(B, 16, N, 64, device='CUDA').realize()
          
          os.environ['FLASH_ATTENTION'] = '1'
          flash_out = q.scaled_dot_product_attention(k, v).numpy()
          
          os.environ['FLASH_ATTENTION'] = '0'
          sdpa_out = q.scaled_dot_product_attention(k, v).numpy()
          
          np.testing.assert_allclose(flash_out, sdpa_out, rtol=1e-2, atol=1e-2)
  
  def test_flash_attention_fallback(self):
    """Test that non-supported configs fall back to standard SDPA."""
    # D != 64 should fallback
    q = Tensor.randn(2, 8, 128, 32, device='CUDA').realize()
    k = Tensor.randn(2, 8, 128, 32, device='CUDA').realize()
    v = Tensor.randn(2, 8, 128, 32, device='CUDA').realize()
    
    os.environ['FLASH_ATTENTION'] = '1'
    # Should not crash, just use standard SDPA
    out = q.scaled_dot_product_attention(k, v).realize()
    self.assertEqual(out.shape, (2, 8, 128, 32))

if __name__ == '__main__':
  unittest.main()
