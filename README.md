# HTM Model for Anomaly Detection on SWaT Dataset

## Overview
We explore the use of TSSE for encoding multivariate contextual patterns by combining SDRs from different channels. The approach:
1. Derives extra features from combinations of continuous channel values in Layer 1
2. Encodes these combinations using TSSE
3. Feeds each combination as a new feature into Layer 2

## Example:
The following example demonstrates combining LIT101 (water tank level) and AIT202 (water quality) sensors.

#### Step 1: Encode Channel Combinations
```bash
python swat_htm.py --stages_channels ^
    P1:LIT101:window=5,sdr_size=1024 ^
    P2:AIT202:window=34,sdr_size=2048
```

#### Step 2: Calculate Anomaly Statistics
```bash
python calc_anomaly_stats.py -sn MC -esn MC --stages_channels ^
    P1:LIT101 ^
    P2:AIT202
```

#### Results
- Successfully detected anomalies: #1, #6, and #21
- Notable finding: Anomaly #6 was undetectable using individual channels (LIT101 or AIT202) but was caught using the combined approach.

## TODO
Explore additional channel combinations:
- Within-stage combinations (P1,...,P6)
- Cross-stage sensor relationships
- Focus on channels with intuitive physical relationships