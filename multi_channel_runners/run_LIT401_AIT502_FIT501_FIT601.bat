@REM LIT401 AIT502 FIT501 FIT601 
@REM LIT401 - water level of RO permeate tank
@REM AIT502 - conductivity/pH of water after chemical dosing 
@REM FIT501 - flow meter of RO permeate transfer
@REM FIT601 - flow meter of RO permeate transfer

set SDR_SIZE_1=512  
set SDR_SIZE_2=512
set SDR_SIZE_3=512
set SDR_SIZE_4=512
set WINDOW_1=1
set WINDOW_2=1
set WINDOW_3=21
set WINDOW_4=34


@REM python swat_htm.py -sbp --MC_encoder_type TSSE --stages_channels ^
@REM     P4:LIT401:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
@REM     P5:AIT502:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
@REM     P5:FIT501:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3% ^
@REM     P6:FIT601:window=%WINDOW_4%,sdr_size=%SDR_SIZE_4%

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
@REM     --MC_encoder_type TSSE  ^
@REM     --stages_channels P4:LIT401 P5:AIT502 P5:FIT501 P6:FIT601


@REM python swat_htm.py -sbp --MC_encoder_type spatial --stages_channels ^
@REM     P4:LIT401:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
@REM     P5:AIT502:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
@REM     P5:FIT501:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3% ^
@REM     P6:FIT601:window=%WINDOW_4%,sdr_size=%SDR_SIZE_4%

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
@REM     --MC_encoder_type spatial  ^
@REM     --stages_channels P4:LIT401 P5:AIT502 P5:FIT501 P6:FIT601


@REM python swat_htm.py -sbp --MC_encoder_type temporal --temporal_buffer_size 3 --stages_channels ^
@REM     P4:LIT401:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
@REM     P5:AIT502:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
@REM     P5:FIT501:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3% ^
@REM     P6:FIT601:window=%WINDOW_4%,sdr_size=%SDR_SIZE_4%

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
@REM     --MC_encoder_type temporal --temporal_buffer_size 3  ^
@REM     --stages_channels P4:LIT401 P5:AIT502 P5:FIT501 P6:FIT601


@REM python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3 --stages_channels ^
@REM     P4:LIT401:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
@REM     P5:AIT502:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
@REM     P5:FIT501:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3% ^
@REM     P6:FIT601:window=%WINDOW_4%,sdr_size=%SDR_SIZE_4%

python calc_anomaly_stats.py -sth 0.35 -sn MC -esn MC -ofa _multi_channel  ^
    --MC_encoder_type combined --temporal_buffer_size 3 --combined_weights 0.7 0.3  ^
    --stages_channels P4:LIT401 P5:AIT502 P5:FIT501 P6:FIT601    
