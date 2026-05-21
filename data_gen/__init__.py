"""
Data generation module for solar radio classifier.
Contains synthesizer and noise utility functions.
"""

from .noise_utils import add_noise_to_signal, generate_quiet_sun_noise
from .synthesizer import (
	generate_no_burst,
	generate_type_1,
	generate_type_2,
	generate_type_3,
	generate_type_4,
	generate_type_5,
	generate_type_6,
	generate_type_7,
	generate_type_8,
	generate_rfi,
)
