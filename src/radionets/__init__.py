"""
raionets - Deep Learning-based imaging in radio interferometry

Licensed under a MIT style license - see LICENSE
"""

from matplotlib import colormaps

from radionets.plotting._puor import PuOr, PuOr_r

from .version import __version__

__all__ = ["__version__"]

colormaps.register(cmap=PuOr)
colormaps.register(cmap=PuOr_r)
