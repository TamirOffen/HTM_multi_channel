@REM PIT501 FIT503 - Stage 5 channel combination
@REM PIT501 - Pressure sensor in RO permeate transfer
@REM FIT503 - Flow meter in RO permeate transfer

set SDR_SIZE_1=1024  
set SDR_SIZE_2=2048
set WINDOW_1=5
set WINDOW_2=21


python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
    P5:PIT501:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P5:FIT503:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type TSSE  ^
    --stages_channels P5:PIT501 P5:FIT503


python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
    P5:PIT501:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P5:FIT503:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type spatial  ^
    --stages_channels P5:PIT501 P5:FIT503


python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 3 --stages_channels ^
    P5:PIT501:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P5:FIT503:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 3  ^
    --stages_channels P5:PIT501 P5:FIT503


python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
    P5:PIT501:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P5:FIT503:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3  ^
    --stages_channels P5:PIT501 P5:FIT503    
