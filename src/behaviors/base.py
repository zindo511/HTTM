from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
from src.tracking.track_manager import TrackHistory


class BaseBehaviorDetector(ABC):
    """
    Interface cơ sở cho các module phát hiện hành vi bất thường.
    """
    def __init__(self, config: Dict[str, Any], alert_manager=None):
        self.config = config
        self.alert_manager = alert_manager
        self.enabled = config.get("enabled", True)

    @abstractmethod
    def process(
        self, 
        track: TrackHistory, 
        current_time: float, 
        frame: np.ndarray, 
        zones: List[Dict[str, Any]]
    ):
        """
        Phân tích hành vi của một track_id tại thời điểm current_time.
        """
        pass
