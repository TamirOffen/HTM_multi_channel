@REM multi channel encoding methods (TSSE, spatial, temporal, combined) for continuous channels in the same stage

@REM stage 2
set SDR_SIZE_1=512  
set SDR_SIZE_2=512
set WINDOW_1=34
set WINDOW_2=13

python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
    P2:AIT202:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P2:FIT201:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type TSSE  ^
    --stages_channels P2:AIT202 P2:FIT201


python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
    P2:AIT202:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P2:FIT201:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type spatial  ^
    --stages_channels P2:AIT202 P2:FIT201


python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 3 --stages_channels ^
    P2:AIT202:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P2:FIT201:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 3  ^
    --stages_channels P2:AIT202 P2:FIT201


python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
    P2:AIT202:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P2:FIT201:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3  ^
    --stages_channels P2:AIT202 P2:FIT201 

@REM stage 3
@REM DPIT301 - pressure sensor in RO permeate transfer
@REM FIT301 - flow meter in RO permeate transfer
@REM LIT301 - water level of UF feed tank
set SDR_SIZE_1=1024  
set SDR_SIZE_2=256
set SDR_SIZE_3=512
set WINDOW_1=21
set WINDOW_2=34
set WINDOW_3=1

python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
    P3:DPIT301:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P3:FIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P3:LIT301:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type TSSE  ^
    --stages_channels P3:DPIT301 P3:FIT301 P3:LIT301

python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
    P3:DPIT301:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P3:FIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P3:LIT301:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type spatial  ^
    --stages_channels P3:DPIT301 P3:FIT301 P3:LIT301

python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 3 --stages_channels ^
    P3:DPIT301:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P3:FIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P3:LIT301:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 3  ^
    --stages_channels P3:DPIT301 P3:FIT301 P3:LIT301

python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
    P3:DPIT301:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P3:FIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P3:LIT301:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3  ^
    --stages_channels P3:DPIT301 P3:FIT301 P3:LIT301  


@REM stage 4
set SDR_SIZE_1=512  
set SDR_SIZE_2=512
set SDR_SIZE_3=512
set WINDOW_1=1
set WINDOW_2=21
set WINDOW_3=1

python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
    P4:AIT402:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P4:LIT401:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type TSSE  ^
    --stages_channels P4:AIT402 P4:FIT401 P4:LIT401


python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
    P4:AIT402:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P4:LIT401:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type spatial  ^
    --stages_channels P4:AIT402 P4:FIT401 P4:LIT401

python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 3 --stages_channels ^
    P4:AIT402:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P4:LIT401:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 3  ^
    --stages_channels P4:AIT402 P4:FIT401 P4:LIT401

python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
    P4:AIT402:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P4:LIT401:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3  ^
    --stages_channels P4:AIT402 P4:FIT401 P4:LIT401  


@REM stage 5
set SDR_SIZE_1=256  
set SDR_SIZE_2=256
set SDR_SIZE_3=256
set SDR_SIZE_4=256
set SDR_SIZE_5=256
set SDR_SIZE_6=256
set SDR_SIZE_7=256
set SDR_SIZE_8=256
set SDR_SIZE_9=256
set WINDOW_1=1
set WINDOW_2=1
set WINDOW_3=34
set WINDOW_4=21
set WINDOW_5=21
set WINDOW_6=21
set WINDOW_7=21
set WINDOW_8=5
set WINDOW_9=8

python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
    P5:AIT501:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P5:AIT502:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P5:AIT503:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3% ^
    P5:AIT504:window=%WINDOW_4%,sdr_size=%SDR_SIZE_4% ^
    P5:FIT501:window=%WINDOW_5%,sdr_size=%SDR_SIZE_5% ^
    P5:FIT503:window=%WINDOW_6%,sdr_size=%SDR_SIZE_6% ^
    P5:FIT504:window=%WINDOW_7%,sdr_size=%SDR_SIZE_7% ^
    P5:PIT501:window=%WINDOW_8%,sdr_size=%SDR_SIZE_8% ^
    P5:PIT503:window=%WINDOW_9%,sdr_size=%SDR_SIZE_9%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type TSSE  ^
    --stages_channels P5:AIT501 P5:AIT502 P5:AIT503 P5:AIT504 P5:FIT501 P5:FIT503 P5:FIT504 P5:PIT501 P5:PIT503


python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
    P5:AIT501:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P5:AIT502:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P5:AIT503:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3% ^
    P5:AIT504:window=%WINDOW_4%,sdr_size=%SDR_SIZE_4% ^
    P5:FIT501:window=%WINDOW_5%,sdr_size=%SDR_SIZE_5% ^
    P5:FIT503:window=%WINDOW_6%,sdr_size=%SDR_SIZE_6% ^
    P5:FIT504:window=%WINDOW_7%,sdr_size=%SDR_SIZE_7% ^
    P5:PIT501:window=%WINDOW_8%,sdr_size=%SDR_SIZE_8% ^
    P5:PIT503:window=%WINDOW_9%,sdr_size=%SDR_SIZE_9%

python calc_anomaly_stats.py -sth 0.35 -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type spatial  ^
    --stages_channels P5:AIT501 P5:AIT502 P5:AIT503 P5:AIT504 P5:FIT501 P5:FIT503 P5:FIT504 P5:PIT501 P5:PIT503

python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 8 --stages_channels ^
    P5:AIT501:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P5:AIT502:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P5:AIT503:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3% ^
    P5:AIT504:window=%WINDOW_4%,sdr_size=%SDR_SIZE_4% ^
    P5:FIT501:window=%WINDOW_5%,sdr_size=%SDR_SIZE_5% ^
    P5:FIT503:window=%WINDOW_6%,sdr_size=%SDR_SIZE_6% ^
    P5:FIT504:window=%WINDOW_7%,sdr_size=%SDR_SIZE_7% ^
    P5:PIT501:window=%WINDOW_8%,sdr_size=%SDR_SIZE_8% ^
    P5:PIT503:window=%WINDOW_9%,sdr_size=%SDR_SIZE_9%

python calc_anomaly_stats.py -sth 0.35 -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 8  ^
    --stages_channels P5:AIT501 P5:AIT502 P5:AIT503 P5:AIT504 P5:FIT501 P5:FIT503 P5:FIT504 P5:PIT501 P5:PIT503

python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 8 --combined_weights 0.7 0.3 --stages_channels ^
    P5:AIT501:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P5:AIT502:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P5:AIT503:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3% ^
    P5:AIT504:window=%WINDOW_4%,sdr_size=%SDR_SIZE_4% ^
    P5:FIT501:window=%WINDOW_5%,sdr_size=%SDR_SIZE_5% ^
    P5:FIT503:window=%WINDOW_6%,sdr_size=%SDR_SIZE_6% ^
    P5:FIT504:window=%WINDOW_7%,sdr_size=%SDR_SIZE_7% ^
    P5:PIT501:window=%WINDOW_8%,sdr_size=%SDR_SIZE_8% ^
    P5:PIT503:window=%WINDOW_9%,sdr_size=%SDR_SIZE_9%

python calc_anomaly_stats.py -sth 0.35 -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type combined --temporal_buffer_size 8 --combined_weights 0.7 0.3  ^
    --stages_channels P5:AIT501 P5:AIT502 P5:AIT503 P5:AIT504 P5:FIT501 P5:FIT503 P5:FIT504 P5:PIT501 P5:PIT503  


