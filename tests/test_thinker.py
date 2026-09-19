import pytest
import torch

from ulm_live.thinker import ULMThinker


def test_thinker_lazy_init() -> None:
    thinker = ULMThinker(
        model_name_or_path="mock/path",
        freeze=True,
        load_pretrained=False,
    )
    assert thinker.model_name_or_path == "mock/path"
    assert thinker.freeze is True
    assert thinker.hidden_layer == -1


def test_thinker_invalid_path_raises_error() -> None:
    with pytest.raises(RuntimeError, match="Failed to load"):
        ULMThinker(
            model_name_or_path="/completely/invalid/path/xyz",
            load_pretrained=True,
        )


# ---------------------------------------------------------
# Integration Tests (executed with: pytest -m integration)
# ---------------------------------------------------------


@pytest.mark.integration
def test_real_thinker_forward_hidden() -> None:
    """Test actual ULM Thinker forward hidden states extraction."""
    thinker = ULMThinker(
        model_name_or_path="Qwen/Qwen3-1.7B",
        device="cpu",
        hidden_layer=-1,
        freeze=True,
        load_pretrained=True,
    )
    assert thinker.hidden_size == 2048
    assert thinker.num_layers == 28

    inputs = thinker.tokenize("울산 날씨 억수로 덥다")
    assert "input_ids" in inputs
    assert inputs["input_ids"].ndim == 2

    hidden = thinker.forward_hidden(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
    assert hidden.ndim == 3
    assert hidden.shape[0] == 1
    assert hidden.shape[1] == inputs["input_ids"].shape[1]
    assert hidden.shape[2] == 2048
