"""Token alignment and explicit text modulation for the conditional pilot."""

import pytest
import torch

from scripts.pilot_linguistic_v4 import AlignedTextFusion


def test_text_modulation_is_aligned_and_starts_from_semantic_baseline():
    model = AlignedTextFusion(vocabulary=8, semantic_dim=4, token_dim=3)
    hidden = torch.randn(2, 5, 4)
    ids = torch.randint(0, 8, (2, 5))
    torch.testing.assert_close(model(hidden, ids), hidden)
    with torch.no_grad():
        model.modulation.weight.fill_(.2)
    changed = model(hidden, ids)
    assert not torch.allclose(changed, hidden)
    with pytest.raises(ValueError, match="align"):
        model(hidden, ids[:, :-1])
