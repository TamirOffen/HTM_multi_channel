@REM LIT401 FIT401 - Stage 4 channel combination
@REM LIT401 - water level of Reverse Osmosis feed tank
@REM FIT401 - outflow of Reverse Osmosis feed tank

set SDR_SIZE_1=512  
set SDR_SIZE_2=1024
set WINDOW_1=1
set WINDOW_2=21


@REM python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
@REM     P4:LIT401:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
@REM     P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
@REM     --MC_encoder_type TSSE  ^
@REM     --stages_channels P4:LIT401 P4:FIT401


@REM python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
@REM     P4:LIT401:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
@REM     P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
@REM     --MC_encoder_type spatial  ^
@REM     --stages_channels P4:LIT401 P4:FIT401


@REM python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 3 --stages_channels ^
@REM     P4:LIT401:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
@REM     P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
@REM     --MC_encoder_type temporal --temporal_buffer_size 3  ^
@REM     --stages_channels P4:LIT401 P4:FIT401


@REM python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
@REM     P4:LIT401:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
@REM     P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
@REM     --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3  ^
@REM     --stages_channels P4:LIT401 P4:FIT401    

@REM for temporal buffer sizes of 2, 5, and 8:
python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 2 --stages_channels ^
    P4:LIT401:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 2  ^
    --stages_channels P4:LIT401 P4:FIT401

python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 5 --stages_channels ^
    P4:LIT401:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 5  ^
    --stages_channels P4:LIT401 P4:FIT401

python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 8 --stages_channels ^
    P4:LIT401:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 8  ^
    --stages_channels P4:LIT401 P4:FIT401

