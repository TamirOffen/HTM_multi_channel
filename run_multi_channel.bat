:: encoding_type = raw, encoding_duration_enabled = False, 
:: sbp = False, hierarchy_lvl = number of channels

@REM python swat_htm.py --stages_channels ^
@REM     P1:LIT101:window=5,sdr_size=1024 ^
@REM     P2:AIT202:window=34,sdr_size=2048

python calc_anomaly_stats.py -sn MC -esn MC --stages_channels ^
    P1:LIT101 ^
    P2:AIT202
    
