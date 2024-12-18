@REM AIT402 FIT401 - Stage 4 channel combination
@REM AIT402 - conductivity of RO feed 
@REM FIT401 - outflow of RO feed tank

set SDR_SIZE_1=256  
set SDR_SIZE_2=1024
set WINDOW_1=1
set WINDOW_2=21


python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
    P4:AIT402:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type TSSE  ^
    --stages_channels P4:AIT402 P4:FIT401


python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
    P4:AIT402:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type spatial  ^
    --stages_channels P4:AIT402 P4:FIT401


python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 3 --stages_channels ^
    P4:AIT402:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 3  ^
    --stages_channels P4:AIT402 P4:FIT401


python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
    P4:AIT402:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3  ^
    --stages_channels P4:AIT402 P4:FIT401    
