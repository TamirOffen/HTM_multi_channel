@REM DPIT301 FIT301 LIT301 - Stage 3 channel combination
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


@REM python swat_htm.py -sbp --MC_encoder_type combined --temporal_buffer_size 2 --combined_weights 0.7 0.3 --stages_channels ^
@REM     P3:DPIT301:window=%WINDOW_1%,sdr_size=%SDR_SIZE_1% ^
@REM     P3:FIT301:window=%WINDOW_2%,sdr_size=%SDR_SIZE_2% ^
@REM     P3:LIT301:window=%WINDOW_3%,sdr_size=%SDR_SIZE_3%

@REM python calc_anomaly_stats.py -sn MC -esn MC -ofa _multi_channel  ^
@REM     --MC_encoder_type combined --temporal_buffer_size 2 --combined_weights 0.7 0.3  ^
@REM     --stages_channels P3:DPIT301 P3:FIT301 P3:LIT301    
