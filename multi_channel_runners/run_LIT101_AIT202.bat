@REM LIT101 AIT202 - Stage 1 and 2 channel combination
@REM LIT101 - raw water level
@REM AIT202 - conductivity/pH of water after chemical dosing 

set SDR_SIZE_1=512  
set SDR_SIZE_2=512
set WINDOW_1=5
set WINDOW_2=34


python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
    P1:LIT101:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P2:AIT202:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type TSSE  ^
    --stages_channels P1:LIT101 P2:AIT202


python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
    P1:LIT101:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P2:AIT202:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type spatial  ^
    --stages_channels P1:LIT101 P2:AIT202


python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 3 --stages_channels ^
    P1:LIT101:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P2:AIT202:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 3  ^
    --stages_channels P1:LIT101 P2:AIT202


python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
    P1:LIT101:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P2:AIT202:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3  ^
    --stages_channels P1:LIT101 P2:AIT202    
