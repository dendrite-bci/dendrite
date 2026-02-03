"""Shared preprocessing utilities.

Functions used by both online and offline preprocessing systems.
"""


def get_valid_sample_rates(original_rate: float) -> list[int]:
    """Get valid target sample rates (integer divisors only).

    Uses integer decimation divisors common in BCI: 1, 2, 4, 5, 8, 10, 16, 20.
    Only includes rates that result in exact integer Hz values >= 64 Hz.

    Args:
        original_rate: Source sample rate in Hz

    Returns:
        List of valid target rates in Hz, sorted descending
    """
    rates = []
    for divisor in [1, 2, 4, 5, 8, 10, 16, 20]:
        rate = original_rate / divisor
        if rate >= 64 and rate == int(rate):  # Min 64 Hz, must be integer
            rates.append(int(rate))
    return sorted(set(rates), reverse=True)
