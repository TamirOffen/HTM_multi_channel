@REM LIT101 AIT202 LIT301 FIT301 - Stage 1, 2 and 3 channel combination
@REM LIT101 - raw water level
@REM AIT202 - conductivity/pH of water after chemical dosing 
@REM LIT301 - water level of UF feed tank
@REM FIT301 - flow meter of RO permeate transfer

set SDR_SIZE_1=1024  
set SDR_SIZE_2=2048
set SDR_SIZE_3=512
set SDR_SIZE_4=512
set WINDOW_1=5
set WINDOW_2=34
set WINDOW_3=1
set WINDOW_4=34


python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
    P1:LIT101:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P2:AIT202:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P3:LIT301:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3% ^
    P3:FIT301:window=%WINDOW_4%,sdr_size=%SDR_SIZE_4%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type TSSE  ^
    --stages_channels P1:LIT101 P2:AIT202 P3:LIT301 P3:FIT301


python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
    P1:LIT101:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P2:AIT202:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P3:LIT301:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3% ^
    P3:FIT301:window=%WINDOW_4%,sdr_size=%SDR_SIZE_4%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type spatial  ^
    --stages_channels P1:LIT101 P2:AIT202 P3:LIT301 P3:FIT301


python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 3 --stages_channels ^
    P1:LIT101:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P2:AIT202:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P3:LIT301:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3% ^
    P3:FIT301:window=%WINDOW_4%,sdr_size=%SDR_SIZE_4%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 3  ^
    --stages_channels P1:LIT101 P2:AIT202 P3:LIT301 P3:FIT301


python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
    P1:LIT101:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P2:AIT202:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P3:LIT301:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3% ^
    P3:FIT301:window=%WINDOW_4%,sdr_size=%SDR_SIZE_4%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3  ^
    --stages_channels P1:LIT101 P2:AIT202 P3:LIT301 P3:FIT301    
