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


def DDos(
    packets: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    rate = kwargs.get("rate", 0.5)
    n_victim = kwargs.get("n_victim", 1)
    logger = logging.getLogger(name="Manipulator")
    success = 0
    all_src = packets["src"].unique()
    time_min = packets["time"].min()
    time_max = packets["time"].max()
    n_injection = int(len(packets) * rate)
    inject_time = np.random.uniform(time_min, time_max, n_injection)
    inject_src = np.random.choice(all_src, n_injection)
    src = packets["src"].value_counts()
    victim_ids = src.index[0:n_victim]
    inject_dst = np.random.choice(victim_ids, n_injection)
    request_packets = packets[packets["message_type"] == "REQUEST"]
    # Choose the first request packet as the injection packet
    injection_packet = request_packets.iloc[[0], :].copy()
    for i in range(n_injection):
        injection_packet["src"] = inject_src[i]
        injection_packet["dst"] = inject_dst[i]
        injection_packet["time"] = inject_time[i]
        packets = pd.concat([packets, injection_packet], ignore_index=True, axis=0)
        success += 1
    packets = packets.sort_values(by=["time"])
    logger.critical(f"Attack finished. Success attack: {success}")
    return packets.reset_index(drop=True)
