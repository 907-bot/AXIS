"""Event streaming module for AXIS — Kafka integration."""
from .producer import EventProducer, KafkaConfig

__all__ = ["EventProducer", "KafkaConfig"]
