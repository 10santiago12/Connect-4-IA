import numpy as np
from connect4.policy import Policy
try:
    from typing import override
except ImportError:  # Python < 3.12
    try:
        from typing_extensions import override
    except ImportError:
        def override(func):
            return func


class OhYes(Policy):

    @override
    def mount(self) -> None:
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        rng = np.random.default_rng()
        available_cols = [c for c in range(7) if s[0, c] == 0]
        return int(rng.choice(available_cols))
