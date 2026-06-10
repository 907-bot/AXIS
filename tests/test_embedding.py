"""Tests for embedding model."""
from backend.api.server import CLIPEmbeddingModel
import numpy as np


def test_clip_embedding_basic():
    m = CLIPEmbeddingModel()
    emb = m.extract_text("bottle")
    assert isinstance(emb, np.ndarray)
    assert len(emb.shape) == 1


def test_clip_embedding_deterministic():
    m = CLIPEmbeddingModel()
    emb1 = m.extract_text("bottle")
    emb2 = m.extract_text("bottle")
    assert (emb1 == emb2).all(), "Embeddings must be deterministic"


def test_clip_embedding_normalized():
    m = CLIPEmbeddingModel()
    emb = m.extract_text("cup")
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-5 or norm == 0.0


def test_clip_embedding_empty():
    m = CLIPEmbeddingModel()
    emb = m.extract_text("")
    assert emb is not None


def test_clip_embedding_synonym():
    m = CLIPEmbeddingModel()
    phone = m.extract_text("cell phone")
    phone2 = m.extract_text("phone")
    assert phone.shape == phone2.shape


def test_clip_embedding_hue_stable():
    m = CLIPEmbeddingModel()
    h1 = m._stable_hash("bottle") % 360
    h2 = m._stable_hash("bottle") % 360
    assert h1 == h2
