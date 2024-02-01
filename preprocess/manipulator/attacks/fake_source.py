"""
Copyright 2023-2024 Shanghai University Cyber Security Laboratary

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pandas as pd
import numpy as np
import logging


def FakeSource(
    packets: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    rate = kwargs.get("rate", 0.8)
    n_repeat = kwargs.get("n_repeat", 2)
    logger = logging.getLogger(name="Manipulator")
    success = 0
    victim_packets = packets.sample(frac=rate)
    # all_ids = packets["src"].unique()
    # fake_id = np.random.choice(all_ids)
    fake_id = "10.0.0.1"
    for _, p in victim_packets.iterrows():
        # Get the src, dst, time of current error packet
        src, dst, time = p[["src", "dst", "time"]]
        # Find the next packet whose src is the same as the chosen packet's src
        next_packet = packets[(packets["src"] == src) & (packets["time"] > time)]
        if len(next_packet) == 0:
            continue
        else:
            next_packet = next_packet.iloc[0]
        for _ in range(n_repeat):
            # Calculate injection time
            inject_time = next_packet["time"] - (
                time - next_packet["time"]
            ) * np.random.uniform(0.1, 0.9)
            # Make an injection packet
            inject_packet = p.copy()
            inject_packet[["src", "dst", "time"]] = [
                fake_id,
                dst,
                inject_time,
            ]
            # Add injection packet to the end of `packets`
            # packets = pd.concat([packets, inject_packet], ignore_index=True, axis=0)
            packets = packets._append(inject_packet)
            logger.info(
                f"Injected a packet at time {inject_time} "
                + f"from {fake_id} to {dst}"
            )
            success += 1
    # Sort the packets by time
    packets = packets.sort_values(by=["time"])
    logger.critical(f"Attack finished. Success attack: {success}")
    return packets.reset_index(drop=True)
