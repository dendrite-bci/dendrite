"""
Plot Managers Package

Contains specialized plot managers for different types of visualizations
in the BMI dashboard.
"""

from .eeg_plots import EEGPlotManager
from .erp_plots import ERPPlotManager
from .event_plots import EventPlotManager
from .modality_plots import ModalityPlotManager
from .neurofeedback_plots import NeurofeedbackPlotManager
from .performance_plots import PerformancePlotManager
from .psd_plots import PSDPlotManager

__all__ = [
    "EEGPlotManager",
    "EventPlotManager",
    "PerformancePlotManager",
    "ModalityPlotManager",
    "NeurofeedbackPlotManager",
    "ERPPlotManager",
    "PSDPlotManager",
]
