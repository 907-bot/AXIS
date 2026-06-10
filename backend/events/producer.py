"""Kafka event producer for AXIS scene events."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


@dataclass
class KafkaConfig:
    """Kafka connection configuration."""
    bootstrap_servers: str = "localhost:9092"
    topic: str = "axis-scene-events"
    client_id: str = "axis-server"
    max_buffer: int = 100


class EventProducer:
    """Produces scene events to Kafka topic.

    Falls back to in-memory buffering if Kafka is not available.
    """

    def __init__(self, config: Optional[KafkaConfig] = None) -> None:
        self.config = config or KafkaConfig()
        self._producer = None
        self._buffer: List[Dict[str, Any]] = []
        self._connected = False
        self._connect()

    def _connect(self) -> None:
        try:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=self.config.client_id,
                value_serializer=lambda v: json.dumps(v, default=str).encode(),
                acks="all",
                retries=3,
                max_block_ms=2000,
            )
            self._connected = True
            logger.info(f"Connected to Kafka at {self.config.bootstrap_servers}")
        except ImportError:
            logger.info("kafka-python not installed, using in-memory event buffer")
        except Exception as e:
            logger.warning(f"Kafka connection failed ({e}), using in-memory buffer")

    def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to Kafka or buffer."""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            **data,
        }
        if self._producer and self._connected:
            try:
                future = self._producer.send(self.config.topic, event)
                future.get(timeout=1)
            except Exception as e:
                logger.warning(f"Kafka send failed: {e}")
                self._buffer_event(event)
        else:
            self._buffer_event(event)

    def _buffer_event(self, event: Dict[str, Any]) -> None:
        self._buffer.append(event)
        if len(self._buffer) > self.config.max_buffer:
            self._buffer.pop(0)

    def flush(self) -> List[Dict[str, Any]]:
        """Flush buffer and return events."""
        events = list(self._buffer)
        self._buffer.clear()
        return events

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent buffered events."""
        return list(self._buffer[-limit:])

    def close(self) -> None:
        if self._producer:
            self._producer.flush()
            self._producer.close()
