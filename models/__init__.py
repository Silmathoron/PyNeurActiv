#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=============
Models module
=============

Module containing theoretical or computational models to study or predict the
behavior of neuronal populations.


Content
=======
"""

from .fardet_2017_synchronous_bursting import Fardet2017_SynchroBurst
from .simulator_synchronous_bursting import Simulator_SynchroBurst

__all__ = [
    "Fardet2017_SynchroBurst",
    "Simulator_SynchroBurst",
]
