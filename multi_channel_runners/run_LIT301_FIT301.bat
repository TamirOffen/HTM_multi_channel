@REM LIT301 FIT301 - Stage 3 channel combination
@REM LIT301 - water level of UF feed tank
@REM FIT301 - outflow of UF feed tank

set SDR_SIZE_1=1024  
set SDR_SIZE_2=512
set WINDOW_1=1
set WINDOW_2=34


python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
    P3:LIT301:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P3:FIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type TSSE  ^
    --stages_channels P3:LIT301 P3:FIT301


python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
    P3:LIT301:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P3:FIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type spatial  ^
    --stages_channels P3:LIT301 P3:FIT301


python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 3 --stages_channels ^
    P3:LIT301:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P3:FIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 3  ^
    --stages_channels P3:LIT301 P3:FIT301


python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
    P3:LIT301:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P3:FIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3  ^
    --stages_channels P3:LIT301 P3:FIT301    
