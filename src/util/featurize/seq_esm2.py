import torch
import torch.nn as nn
from numpy.typing import NDArray
from typing import List


class esm_embedding(nn.Module):
    """
    ProtT5-XL sequence encoder.

    (Class name kept as ``esm_embedding`` for backward compatibility with
    existing checkpoints — the actual encoder loaded is ProtT5-XL via
    HuggingFace ``T5EncoderModel``.)

    Two modes:
      - Precomputed: read embeddings from LMDB (no model loaded, saves GPU memory).
      - Realtime:    load T5EncoderModel and run forward pass.
    """
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        config: dict = None
    ):
        super().__init__()

        if config is None:
            config = {}

        self.model_name = "prot_t5_xl"
        self.embed_dim  = 1024
        self.num_layers = 24

        # Layer-fusion weights (kept in state_dict for both modes for consistency).
        self.num_fusion_layers = 4
        self.layer_weights = nn.Parameter(torch.ones(self.num_fusion_layers))

        # Precomputed-only mode: skip loading model weights.
        precomputed_only = config.get("precomputed_only", False)
        if precomputed_only:
            self.device = device
            self.model = None
            self.tokenizer = None
            print(f"[SeqEncoder] ProtT5-XL — precomputed mode (dim={self.embed_dim})")
            return

        # Realtime mode: load ProtT5 encoder.
        from transformers import T5EncoderModel, T5Tokenizer
        model_path = config.get("prot5_model_path", "Rostlab/prot_t5_xl_uniref50")
        print(f"[SeqEncoder] Loading ProtT5-XL from {model_path} ...")

        self.tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.model = self.model.to(device)
        self.device = device

        # Encoder is always frozen — gradients flow only through downstream modules.
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[SeqEncoder] ProtT5-XL loaded — {self.num_layers} layers, dim={self.embed_dim}")

    # ------------------------------------------------------------------
    # Tokenization helper
    # ------------------------------------------------------------------
    def _tokenize(self, seqs: List[str]):
        """Tokenize sequences for ProtT5 (space-separated AAs, replace non-standard)."""
        def _sanitize(aa):
            return "X" if aa in ("U", "Z", "O", "B") else aa

        spaced = [" ".join(_sanitize(aa) for aa in seq) for seq in seqs]
        encoding = self.tokenizer(
            spaced, add_special_tokens=True, padding="longest",
            return_tensors="pt"
        )
        return encoding["input_ids"], encoding["attention_mask"]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, seq: str) -> torch.Tensor:
        """Encode a single sequence, returns (seq_len, 1024)."""
        embs = self.forward_batch([seq])
        return embs[0]

    def forward_batch(self, seqs: List[str]) -> List[torch.Tensor]:
        """
        Encode a batch of sequences via ProtT5 encoder.
        Returns list of (seq_len, 1024) tensors.
        """
        input_ids, attention_mask = self._tokenize(seqs)
        current_device = next(self.model.parameters()).device
        input_ids = input_ids.to(current_device)
        attention_mask = attention_mask.to(current_device)

        token_representations = self._forward_standard(input_ids, attention_mask)

        # Slice out per-sequence embeddings (exclude EOS token).
        embeddings = []
        for i, seq in enumerate(seqs):
            real_tokens = int(attention_mask[i].sum().item()) - 1
            seq_len = min(len(seq), real_tokens, token_representations.shape[1])
            emb = token_representations[i, :seq_len]
            embeddings.append(emb)
        return embeddings

    def _forward_standard(self, input_ids, attention_mask):
        """Run the encoder, fuse last N hidden states, apply final layer norm."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        last_n  = hidden_states[-self.num_fusion_layers:]
        weights = torch.softmax(self.layer_weights, dim=0)
        fused   = sum(w * h for w, h in zip(weights, last_n))

        if hasattr(self.model.encoder, 'final_layer_norm') \
                and self.model.encoder.final_layer_norm is not None:
            fused = self.model.encoder.final_layer_norm(fused)

        return fused

    @torch.no_grad()
    def get_embedding(self, seq: str) -> NDArray:
        self.eval()
        return self.forward(seq).cpu().numpy()

    @property
    def embedding_dim(self):
        return self.embed_dim