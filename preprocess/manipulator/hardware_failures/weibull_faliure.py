import pandas as pd
import numpy as np
import logging

from . import weibull_distribution as wd


def WeibullFailure(
    packets: pd.DataFrame,
    **kwargs,
):
    n_victim = kwargs.get("n_victim", 2)
    params = [[1, 2], [1, 1]]
    alpha, beta = params[np.random.randint(0, len(params))]
    logger = logging.getLogger(name="Manipulator")
    success = 0
    # Choose the victims
    start_time = packets["time"].min()
    times = packets["time"] - start_time
    times = wd.SCALAR.fit_transform(times.to_numpy().reshape(-1, 1))
    # Get the index of victim packets
    victim_packet_ilocs = packets.index
    # Drop the victim packets
    for iloc in victim_packet_ilocs:
        if np.random.uniform() - 0.2 < wd.distribution(
            alpha, beta, times[iloc]
        ):
            packets = packets.drop(iloc)
            success += 1
    logger.critical(f"Attack finished. Success dropped: {success}")
    return packets.reset_index(drop=True)
