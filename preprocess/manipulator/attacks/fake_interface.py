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


def FakeInterface(packets: pd.DataFrame, **kwargs) -> pd.DataFrame:
    rate = kwargs.get("rate", 0.8)
    logger = logging.getLogger(name="Manipulator")
    success = 0
    inject_packets = packets.sample(frac=rate)
    packets = packets.drop(inject_packets.index)
    inject_packets["interface"] =  np.random.choice([2,3,4,5,6], len(inject_packets))
    success = len(inject_packets)
    packets = pd.concat([packets, inject_packets], ignore_index=True)
    packets = packets.sort_values(by=["time"])
    logger.critical(f"Attack finished. Success attack: {success}")
    return packets.reset_index(drop=True)
