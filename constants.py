from typing import Literal

SamplerType = Literal['lognormal', 'weibull']
ScoreFNType = Literal['dot', 'bilinear', 'concat', 'hadamard']
SimplexFNType = Literal['linear', 'exp']