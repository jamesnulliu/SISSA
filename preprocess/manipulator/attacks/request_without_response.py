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
import logging


def RequestWithoutResponse(
    packets: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    rate = kwargs.get("rate", 0.8)
    n_victim = kwargs.get("n_victim", 2)
    logger = logging.getLogger(name="Manipulator")
    # Choose the response packets
    response_packets = packets[packets["message_type"] == "RESPONSE"].sample(frac=rate)
    # Drop the chosen response packets from `packets`
    packets = packets.drop(response_packets.index)
    # Check for nan
    logger.critical(f"Attack finished. Success attack: {len(response_packets)}")
    return packets.reset_index(drop=True)
