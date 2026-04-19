import torch
import unittest
from src.losses import PartialFocalCELoss 

class TestLosses(unittest.TestCase):
    def setUp(self):
        self.criterion = PartialFocalCELoss(gamma=2.0, ignore_index=-1)

    def test_pfce_ignore_index(self):
        """Verify that pixels marked as ignore_index contribute zero to the loss."""
        logits = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.full((2, 32, 32), -1, dtype=torch.long)
        
        loss = self.criterion(logits, target)
        self.assertEqual(loss.item(), 0.0, "Loss should be 0 when all pixels are ignored.")
        
    def test_pfce_calculation(self):
        """Verify that high confidence in the correct class results in a low loss."""
        # Create logits where the first pixel is heavily biased towards Class 1
        logits = torch.zeros(1, 2, 4, 4)
        logits[0, 1, 0, 0] = 50.0  # Extremely high confidence for Class 1 at (0,0)
        
        # Target has only one labeled point at (0,0) with Class 1
        target = torch.full((1, 4, 4), -1, dtype=torch.long)
        target[0, 0, 0] = 1 
        
        loss = self.criterion(logits, target)
        

        # at high confidence.
        self.assertLess(loss.item(), 0.2, f"Loss was {loss.item()}, expected < 0.2")

# Updated Loader logic to avoid DeprecationWarning
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLosses)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
    