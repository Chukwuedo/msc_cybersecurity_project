"""
Feature mapping definitions for dataset harmonization.

This module contains the mapping definitions between CICIOT2023 and CICDIAD2024
datasets for feature harmonization.
"""

# Feature mapping table: (common_name, ciciot_features, cicdiad_features, mapping_type, rationale)
FEATURE_MAPPING_DATA = [
    (
        "flow_duration",
        "['flow_duration']",
        "['Flow Duration']",
        "direct",
        "Same concept in both datasets: total duration of the flow/window.",
    ),
    (
        "flow_packets_per_second",
        "['Rate']",
        "['Flow Packets/s']",
        "direct",
        "Packets per second over the whole flow.",
    ),
    (
        "forward_packets_per_second",
        "['Srate']",
        "['Fwd Packets/s']",
        "direct",
        "Forward-direction packet rate.",
    ),
    (
        "backward_packets_per_second",
        "['Drate']",
        "['Bwd Packets/s']",
        "direct",
        "Backward-direction packet rate.",
    ),
    (
        "flow_iat_mean",
        "['IAT']",
        "['Flow IAT Mean']",
        "direct",
        "Mean inter-arrival time across packets in the flow.",
    ),
    (
        "packet_length_min",
        "['Min']",
        "['Packet Length Min']",
        "direct",
        "Minimum packet length observed in the flow.",
    ),
    (
        "packet_length_max",
        "['Max']",
        "['Packet Length Max']",
        "direct",
        "Maximum packet length observed in the flow.",
    ),
    (
        "packet_length_mean",
        "['AVG']",
        "['Packet Length Mean']",
        "direct",
        "Average packet length across the flow.",
    ),
    (
        "packet_length_std",
        "['Std']",
        "['Packet Length Std']",
        "direct",
        "Standard deviation of packet length across the flow.",
    ),
    (
        "packet_length_range",
        "['Max - Min']",
        "['Packet Length Max - Packet Length Min']",
        "engineered",
        "Range computed as max - min in both datasets.",
    ),
    (
        "fin_flag_count",
        "['fin_flag_number', 'fin_count']",
        "['FIN Flag Count']",
        "direct",
        "FIN occurrences across the flow.",
    ),
    (
        "syn_flag_count",
        "['syn_flag_number', 'syn_count']",
        "['SYN Flag Count']",
        "direct",
        "SYN occurrences across the flow.",
    ),
    (
        "rst_flag_count",
        "['rst_flag_number', 'rst_count']",
        "['RST Flag Count']",
        "direct",
        "RST occurrences across the flow.",
    ),
    (
        "psh_flag_count",
        "['psh_flag_number']",
        "['PSH Flag Count']",
        "direct",
        "PSH occurrences across the flow.",
    ),
    (
        "ack_flag_count",
        "['ack_flag_number', 'ack_count']",
        "['ACK Flag Count']",
        "direct",
        "ACK occurrences across the flow.",
    ),
    (
        "ece_flag_count",
        "['ece_flag_number']",
        "['ECE Flag Count']",
        "direct",
        "ECE occurrences across the flow.",
    ),
    (
        "cwr_flag_count",
        "['cwr_flag_number']",
        "['CWR Flag Count']",
        "direct",
        "CWR occurrences across the flow.",
    ),
    (
        "urg_flag_count",
        "['urg_count']",
        "['URG Flag Count']",
        "direct",
        "URG occurrences across the flow.",
    ),
    (
        "total_packets",
        "['Number']",
        "['Total Fwd Packet + Total Bwd packets']",
        "composite_sum",
        "Total packets over both directions.",
    ),
    (
        "total_bytes",
        "['Tot sum']",
        "['Total Length of Fwd Packet + Total Length of Bwd Packet']",
        "composite_sum",
        "Total bytes over both directions.",
    ),
    (
        "average_packet_size",
        "['Tot size']",
        "['Average Packet Size']",
        "direct_or_equivalent",
        "Mean packet size over the flow.",
    ),
    (
        "flow_bytes_per_second",
        "['tot_sum / flow_duration']",
        "['Flow Bytes/s']",
        "engineered",
        "Flow bytes per second, computed or direct.",
    ),
    (
        "header_length_total",
        "['header_len']",
        "['Fwd Header Length + Bwd Header Length']",
        "composite_sum",
        "Total header length over both directions.",
    ),
]


def get_feature_mapping_df():
    """Return feature mappings as a Polars DataFrame."""
    import polars as pl

    return pl.DataFrame(
        {
            "Common Feature": [item[0] for item in FEATURE_MAPPING_DATA],
            "CICIOT2023 Features": [item[1] for item in FEATURE_MAPPING_DATA],
            "CICDIAD2024 Features": [item[2] for item in FEATURE_MAPPING_DATA],
            "Mapping Type": [item[3] for item in FEATURE_MAPPING_DATA],
            "Rationale": [item[4] for item in FEATURE_MAPPING_DATA],
        }
    )


def get_mapping_statistics():
    """Return statistics about feature mappings."""
    direct_count = len([item for item in FEATURE_MAPPING_DATA if item[3] == "direct"])
    composite_count = len(
        [item for item in FEATURE_MAPPING_DATA if item[3] == "composite_sum"]
    )
    engineered_count = len(
        [item for item in FEATURE_MAPPING_DATA if item[3] == "engineered"]
    )

    return {
        "total_features": len(FEATURE_MAPPING_DATA),
        "direct_mappings": direct_count,
        "composite_mappings": composite_count,
        "engineered_mappings": engineered_count,
    }
