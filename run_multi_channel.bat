:: encoding_type = raw, encoding_duration_enabled = False, 
:: sbp = False, hierarchy_lvl = number of channels
:: order of channels matters!!!

@REM python swat_htm.py --stages_channels ^
@REM     P1:LIT101:window=5,sdr_size=1024 ^
@REM     P2:AIT202:window=34,sdr_size=2048

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel --stages_channels ^
    P1:LIT101 ^
    P2:AIT202
    

@REM python swat_htm.py --stages_channels ^
@REM     P5:FIT503:window=21,sdr_size=2048 ^
@REM     P5:PIT501:window=5,sdr_size=1024 ^
@REM     P5:PIT503:window=8,sdr_size=1024

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel --stages_channels ^
    P5:FIT503 ^
    P5:PIT501 ^
    P5:PIT503


@REM python swat_htm.py --stages_channels ^
@REM     P5:FIT501:window=21,sdr_size=1024 ^
@REM     P5:PIT501:window=5,sdr_size=1024 ^
@REM     P5:AIT503:window=34,sdr_size=512

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel --stages_channels ^
    P5:FIT501 ^
    P5:PIT501 ^
    P5:AIT503


@REM python swat_htm.py --stages_channels ^
@REM     P3:DPIT301:window=21,sdr_size=512 ^
@REM     P3:FIT301:window=34,sdr_size=512 ^
@REM     P3:LIT301:window=1,sdr_size=512

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel --stages_channels ^
    P3:DPIT301 ^
    P3:FIT301 ^
    P3:LIT301   

python calc_anomaly_stats.py -ofa _multi_channel --final_stage 
