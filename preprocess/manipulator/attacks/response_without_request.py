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


def ResponseWithoutRequest(
    packets: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    rate = kwargs.get("rate", 0.8)
    n_victim = kwargs.get("n_victim", 6)
    logger = logging.getLogger(name="Manipulator")
    packets = packets.sort_values(by=["time"])
    request_packets = packets[packets["message_type"] == "REQUEST"].sample(frac=rate)
    packets = packets.drop(request_packets.index)
    logger.critical(f"Attack finished. Success attack: {len(request_packets)}")
    return packets.reset_index(drop=True)
