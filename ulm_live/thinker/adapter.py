from pathlib import Path
from typing import Any
import torch


def _resolve_torch_dtype(dtype_str: str) -> torch.dtype | None:
    if dtype_str == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp16", "float16"):
        return torch.float16
    if dtype_str in ("fp32", "float32"):
        return torch.float32
    return None


class ULMThinker:
    """Adapter for ULM-1.7B language model (Thinker) to extract semantic hidden states."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-1.7B",
        device: str = "auto",
        torch_dtype: str = "auto",
        hidden_layer: int = -1,
        freeze: bool = True,
        load_pretrained: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device_str = device
        self.torch_dtype_str = torch_dtype
        self.hidden_layer = hidden_layer
        self.freeze = freeze

        self._device = self._resolve_device(device)
        self._dtype = _resolve_torch_dtype(torch_dtype)

        self._tokenizer: Any = None
        self._model: Any = None

        if load_pretrained:
            self.load()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return target

    def load(self) -> None:
        """Load tokenizer and model weights from Transformers."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as err:
            raise ImportError("transformers is required to load ULMThinker.") from err

        model_path = Path(self.model_name_or_path)
        path_str = str(model_path) if model_path.exists() else self.model_name_or_path

        # 1. Load Tokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path_str, trust_remote_code=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        except Exception as err:
            raise RuntimeError(f"Failed to load tokenizer from '{path_str}': {err}") from err

        # 2. Check if path is a PEFT LoRA adapter
        is_peft = (model_path / "adapter_config.json").is_file() if model_path.is_dir() else False

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self._dtype is not None and self._device.type != "cpu":
            load_kwargs["torch_dtype"] = self._dtype

        try:
            if is_peft:
                import json
                with open(model_path / "adapter_config.json", "r", encoding="utf-8") as f:
                    adapter_cfg = json.load(f)
                base_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen3-1.7B")
                base_model = AutoModel.from_pretrained(base_name, **load_kwargs)
                try:
                    from peft import PeftModel
                    self._model = PeftModel.from_pretrained(base_model, path_str)
                except ImportError:
                    # If peft not installed, fallback to base model
                    self._model = base_model
            else:
                self._model = AutoModel.from_pretrained(path_str, **load_kwargs)

            self._model.to(self._device)

            if self.freeze:
                for param in self._model.parameters():
                    param.requires_grad = False
                self._model.eval()

        except Exception as err:
            raise RuntimeError(f"Failed to load model from '{path_str}': {err}") from err

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    @property
    def hidden_size(self) -> int:
        if self._model is not None and hasattr(self._model, "config"):
            return int(getattr(self._model.config, "hidden_size", 2048))
        return 2048

    @property
    def num_layers(self) -> int:
        if self._model is not None and hasattr(self._model, "config"):
            return int(getattr(self._model.config, "num_hidden_layers", 28))
        return 28

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer)
        if self._model is not None and hasattr(self._model, "config"):
            return int(getattr(self._model.config, "vocab_size", 151936))
        return 151936

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        if self._model is not None and hasattr(self._model, "dtype"):
            return self._model.dtype
        return self._dtype if self._dtype is not None else torch.float32

    def tokenize(
        self,
        text: str | list[str],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Tokenize text into model input tensors."""
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return {k: v.to(self._device) for k, v in encoded.items()}

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_layer: int | None = None,
    ) -> torch.Tensor:
        """Extract semantic hidden states from the language model.

        Args:
            input_ids: Tensor of token ids (batch_size, sequence_length).
            attention_mask: Optional attention mask.
            hidden_layer: Specific layer index (defaults to self.hidden_layer, e.g. -1 for last layer).

        Returns:
            Tensor of shape (batch_size, sequence_length, hidden_size).
        """
        layer_idx = hidden_layer if hidden_layer is not None else self.hidden_layer
        inp = input_ids.to(self._device)
        mask = attention_mask.to(self._device) if attention_mask is not None else None

        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            outputs = self.model(
                input_ids=inp,
                attention_mask=mask,
                output_hidden_states=True,
            )

        if not hasattr(outputs, "hidden_states") or not outputs.hidden_states:
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            raise RuntimeError("Model outputs do not contain hidden_states.")

        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states)
        # Normalize negative indexing
        idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
        if not (0 <= idx < num_layers):
            raise IndexError(f"hidden_layer index {layer_idx} out of range for {num_layers} layers.")

        return hidden_states[idx]

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate response text for high-level dialogue testing."""
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as err:
            raise ImportError("transformers is required for generate_text.") from err

        inputs = self.tokenize(prompt)
        # If current model is not causal LM, generate is not supported
        if not hasattr(self.model, "generate"):
            return prompt

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)
