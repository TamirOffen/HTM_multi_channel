@REM FIT201 FIT401 FIT501 - 3 different channel combination
@REM FIT201 - Flow meter after chemical dosing
@REM FIT401 - Flow meter for RO feed
@REM FIT501 - Flow meter in RO permeate transfer

set SDR_SIZE_1=512  
set SDR_SIZE_2=512
set SDR_SIZE_3=512
set WINDOW_1=13
set WINDOW_2=21
set WINDOW_3=21


python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
    P2:FIT201:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P5:FIT501:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type TSSE  ^
    --stages_channels P2:FIT201 P4:FIT401 P5:FIT501


python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
    P2:FIT201:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P5:FIT501:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type spatial  ^
    --stages_channels P2:FIT201 P4:FIT401 P5:FIT501


python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 3 --stages_channels ^
    P2:FIT201:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P5:FIT501:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 3  ^
    --stages_channels P2:FIT201 P4:FIT401 P5:FIT501


python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
    P2:FIT201:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P5:FIT501:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3  ^
    --stages_channels P2:FIT201 P4:FIT401 P5:FIT501    



@REM for temporal buffer sizes of 2, 5, and 8:
python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 2 --stages_channels ^
    P2:FIT201:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P5:FIT501:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 2  ^
    --stages_channels P2:FIT201 P4:FIT401 P5:FIT501

python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 5 --stages_channels ^
    P2:FIT201:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P5:FIT501:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 5  ^
    --stages_channels P2:FIT201 P4:FIT401 P5:FIT501

python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 8 --stages_channels ^
    P2:FIT201:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
    P4:FIT401:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
    P5:FIT501:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type temporal --temporal_buffer_size 8  ^
    --stages_channels P2:FIT201 P4:FIT401 P5:FIT501

