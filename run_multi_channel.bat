:: encoding_type = raw, encoding_duration_enabled = False, 
:: sbp = False, hierarchy_lvl = number of channels
:: order of channels matters!!!


@REM python swat_htm.py --MC_encoder_type spatial --stages_channels ^
@REM     P1:LIT101:window=5 ^
@REM     P2:AIT202:window=34

python swat_htm.py --MC_encoder_type combined --temporal_buffer_size 2 --combined_weights 0.7 0.3 --stages_channels ^
    P3:DPIT301:window=21 ^
    P3:FIT301:window=34 ^
    P3:LIT301:window=1

@REM python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 5 --combined_weights 0.7 0.3 --stages_channels ^
@REM     P1:LIT101:window=5,sdr_size=512 ^
@REM     P2:AIT202:window=34,sdr_size=512

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel --stages_channels ^
@REM     P1:LIT101 ^
@REM     P2:AIT202
    

@REM python swat_htm.py --stages_channels ^
@REM     P5:FIT503:window=21,sdr_size=2048 ^
@REM     P5:PIT501:window=5,sdr_size=1024 ^
@REM     P5:PIT503:window=8,sdr_size=1024

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel --stages_channels ^
@REM     P5:FIT503 ^
@REM     P5:PIT501 ^
@REM     P5:PIT503


@REM python swat_htm.py --stages_channels ^
@REM     P5:FIT501:window=21,sdr_size=1024 ^
@REM     P5:PIT501:window=5,sdr_size=1024 ^
@REM     P5:AIT503:window=34,sdr_size=512

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel --stages_channels ^
@REM     P5:FIT501 ^
@REM     P5:PIT501 ^
@REM     P5:AIT503


@REM python swat_htm.py --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
@REM     P3:DPIT301:window=21,sdr_size=512 ^
@REM     P3:FIT301:window=34,sdr_size=256 ^
@REM     P3:LIT301:window=1,sdr_size=512

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel --stages_channels ^
@REM     P3:DPIT301 ^
@REM     P3:FIT301 ^
@REM     P3:LIT301   

@REM python swat_htm.py --MC_encoder_type concat --stages_channels ^
@REM     P1:LIT101:window=5,sdr_size=1024 ^
@REM     P2:FIT201:window=13,sdr_size=1024 ^
@REM     P3:DPIT301:window=21,sdr_size=512 ^
@REM     P4:FIT401:window=21,sdr_size=1024 ^
@REM     P5:AIT504:window=21,sdr_size=1024

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel -sth 0.15 --stages_channels ^
@REM     P1:LIT101 ^
@REM     P2:FIT201 ^
@REM     P3:DPIT301 ^
@REM     P4:FIT401 ^
@REM     P5:AIT504

@REM python swat_htm.py --MC_encoder_type TSSE --stages_channels ^
@REM     P1:LIT101:window=5,sdr_size=1024 ^
@REM     P2:FIT201:window=13,sdr_size=1024 ^
@REM     P4:FIT401:window=21,sdr_size=1024 ^
@REM     P5:AIT504:window=21,sdr_size=1024

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel --stages_channels ^
@REM     P1:LIT101 ^
@REM     P2:FIT201 ^
@REM     P4:FIT401 ^
@REM     P5:AIT504


@REM python calc_anomaly_stats.py -ofa _multi_channel --final_stage 
