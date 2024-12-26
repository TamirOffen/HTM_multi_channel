@REM LIT101 LIT301 LIT401 - Cross stage level indicators combination
@REM LIT101 - raw water tank level
@REM LIT301 - water level in UF feed tank
@REM LIT401 - water level in RO feed tank

set SDR_SIZE_1=512  
set SDR_SIZE_2=512
set SDR_SIZE_3=512
set WINDOW_1=5
set WINDOW_2=1
set WINDOW_3=1


python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
    P1:LIT101:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P3:LIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P4:LIT401:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type TSSE  ^
    --stages_channels P1:LIT101 P3:LIT301 P4:LIT401


python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
    P1:LIT101:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P3:LIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P4:LIT401:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type spatial  ^
    --stages_channels P1:LIT101 P3:LIT301 P4:LIT401


python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 3 --stages_channels ^
    P1:LIT101:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P3:LIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P4:LIT401:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 3  ^
    --stages_channels P1:LIT101 P3:LIT301 P4:LIT401


python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
    P1:LIT101:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P3:LIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P4:LIT401:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3  ^
    --stages_channels P1:LIT101 P3:LIT301 P4:LIT401    
