"""Feature extractors for semantic understanding."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# Placeholder for actual model imports
try:
    import clip
    from transformers import AutoModel, AutoImageProcessor
except ImportError:
    clip = None
    AutoModel = None
    AutoImageProcessor = None


@dataclass
class EmbeddingResult:
    """Feature embedding result."""
    embeddings: np.ndarray  # N, D
    shape: tuple
    model_name: str
    device: str


class FeatureExtractor:
    """Base class for feature extraction."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = None

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract features from image."""
        raise NotImplementedError

    def to(self, device: str):
        """Move model to device."""
        self.device = torch.device(device)
        if self.model:
            self.model.to(self.device)
        return self


class CLIPEmbedder(FeatureExtractor):
    """
    CLIP-based image and text embedding.
    
    Provides:
    - Image embeddings for similarity search
    - Text embeddings for semantic queries
    - Zero-shot classification
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(device)
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load CLIP model."""
        if clip is None:
            logger.warning("CLIP not available - install openai/clip")
            return

        try:
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            logger.info(f"Loaded CLIP: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            self.model = None

    def extract_image(self, image: np.ndarray) -> np.ndarray:
        """Extract image embedding."""
        if self.model is None:
            return np.random.randn(512).astype(np.float32)

        # Preprocess
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image_input)

        return features.cpu().numpy()[0]

    def extract_text(self, text: str) -> np.ndarray:
        """Extract text embedding."""
        if self.model is None:
            return np.random.randn(512).astype(np.float32)

        text_input = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(text_input)

        return features.cpu().numpy()[0]

    def extract_images_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for batch of images."""
        if self.model is None or len(images) == 0:
            return np.random.randn(len(images), 512).astype(np.float32)

        # Preprocess batch
        batch = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(batch)

        return features.cpu().numpy()

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ))

    def text_to_image_score(self, text: str, image: np.ndarray) -> float:
        """Score how well text matches image."""
        text_emb = self.extract_text(text)
        image_emb = self.extract_image(image)
        return self.similarity(text_emb, image_emb)

    def zero_shot_classify(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> List[float]:
        """Classify image using zero-shot CLIP."""
        if self.model is None:
            return [1.0 / len(class_names)] * len(class_names)

        # Get image embedding
        image_features = self.extract_image(image).reshape(1, -1)
        image_features /= np.linalg.norm(image_features, axis=1, keepdims=True)

        # Get text embeddings
        text_inputs = clip.tokenize(class_names).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        text_features = text_features.cpu().numpy()
        text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)

        # Calculate similarities
        similarity = (image_features @ text_features.T)[0]
        probs = np.exp(similarity) / np.sum(np.exp(similarity))
        
        return probs.tolist()


class DINOv2Extractor(FeatureExtractor):
    """
    DINOv2 feature extractor for self-supervised visual features.
    
    Provides:
    - High-quality visual features without labels
    - Patch-level and image-level embeddings
    - Good for retrieval and recognition
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(device)
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load DINOv2 model."""
        if AutoModel is None:
            logger.warning("Transformers not available")
            return

        try:
            self.model = AutoModel.from_pretrained(
                f"facebook/{self.model_name}"
            ).to(self.device)
            self.model.eval()

            # Load processor
            self.processor = AutoImageProcessor.from_pretrained(
                f"facebook/{self.model_name}"
            )

            logger.info(f"Loaded DINOv2: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load DINOv2: {e}")
            self.model = None

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract DINOv2 features from image."""
        if self.model is None:
            return np.random.randn(1024).astype(np.float32)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use CLS token features
        features = outputs.last_hidden_state[:, 0]

        return features.cpu().numpy()[0]

    def extract_patch_features(self, image: np.ndarray) -> np.ndarray:
        """Extract per-patch features for spatial understanding."""
        if self.model is None:
            h, w = image.shape[:2]
            return np.random.randn(h // 16, w // 16, 1024).astype(np.float32)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Reshape to spatial grid
        # DINOv2 outputs sequence of patches
        features = outputs.last_hidden_state[:, 1:]  # Remove CLS

        return features.cpu().numpy()

    def extract_region(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        """Extract features for image region."""
        x, y, w, h = bbox
        region = image[y:y+h, x:x+w]
        return self.extract(region)


class CombinedFeatureExtractor:
    """Combined CLIP + DINOv2 feature extraction."""

    def __init__(
        self,
        clip_model: str = "ViT-L/14",
        dino_model: str = "dinov2_vitl14",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.clip = CLIPEmbedder(clip_model, device)
        self.dino = DINOv2Extractor(dino_model, device)

    def extract(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract both CLIP and DINO features."""
        return {
            "clip": self.clip.extract_image(image),
            "dino": self.dino.extract(image)
        }

    def extract_combined(self, image: np.ndarray) -> np.ndarray:
        """Extract and concatenate features."""
        features = self.extract(image)
        combined = np.concatenate([features["clip"], features["dino"]])
        return combined

    def search_by_text(self, query: str, image: np.ndarray) -> float:
        """Semantic search score."""
        return self.clip.text_to_image_score(query, image)