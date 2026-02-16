from .wavetracker import calc_wavefronts, WaveTrackerOptions, display_model, BasisModel
try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
