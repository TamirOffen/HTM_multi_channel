import sys
import argparse
import json
import csv
import os
import numpy as np
import random
import math
import pandas
import swat_utils
from enum import Enum

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor
import collections
import white_black_list

parser = argparse.ArgumentParser(description='runtime configuration for HTM anomaly detection on SWAT')
parser.add_argument('--stage_name', '-sn', metavar='STAGE_NAME', default='P1',
                    choices=['P1', 'P2', 'P3', 'P4', 'P5', 'P6'], type=str.upper)
parser.add_argument('--channel_name', '-cn', metavar='CHANNEL_NAME', default='LIT101')
parser.add_argument('--channel_type', '-ctype', metavar='CHANNEL_TYPE', default=0, type=int,
                    help='set type 0 for analog, 1 for discrete')
parser.add_argument('--freeze_type', '-ft', default='off', choices=['off', 'during_training', 'end_training'],
                    type=str.lower)
parser.add_argument('--learn_type', '-lt', default='always', choices=['always', 'train_only'], type=str.lower)
parser.add_argument('--sdr_size', '-size', metavar='SDR_SIZE', default=1024, type=int)
parser.add_argument('--connection_segments_gap', '-csg', default=1, type=int)
parser.add_argument('--sdr_sparsity', '-sparsity', metavar='SDR_SPARCITY', default=0.02, type=float)
parser.add_argument('--window', '-w', metavar='MOVMEAN_WINDOW', default=1, type=int)
parser.add_argument('--window_tight', '-wt', default=True, action='store_false')
parser.add_argument('--diff_enabled', '-diff', default=False, action='store_true')
parser.add_argument('--search_best_parameters', '-sbp', default=True, action='store_false')
parser.add_argument('--custom_min', '-cmin', metavar='MIN_VAL', default=2, type=int)
parser.add_argument('--custom_max', '-cmax', metavar='MAX_VAL', default=3, type=int)
# parser.add_argument('--sum_window', '-sw', default=119, type=int, help="moving sum anomaly score window")
parser.add_argument('--sum_window', '-sw', default=120, type=int, help="moving sum anomaly score window")
parser.add_argument('--sum_threshold', '-sth', default=0.6, type=float, help="moving sum anomaly score threshold")
parser.add_argument('--limits_enabled', '-le', default=False, action='store_true')
parser.add_argument('--encoding_duration_enabled', '-ede', default=True)
parser.add_argument('--encoding_duration_value', '-ed_val', default=0, type=int)
parser.add_argument('--encoding_duration_bins', '-ed_bins', default=10, type=int)
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--prefix', default="", type=str.lower)
parser.add_argument('--input_file_path', default="./HTM_input/", type=str)
parser.add_argument('--output_file_path', default="./HTM_results/", type=str)
parser.add_argument('--override_parameters', '-op', default="", type=str,
                    help="override parameter values, group_name,var_name,val,res/../.. ,param value = val/res")
parser.add_argument('--replay_buffer', '-rpb', default=0, type=int)
parser.add_argument('--encoding_type', '-et', metavar='ENCODING_TYPE', default='diff', choices=['raw', 'diff'],
                    type=str.lower)
parser.add_argument('--sampling', '-sg', default=1, type=int, help="sampling interval")
parser.add_argument('--hierarchy_enabled', '-he', default=True, action='store_true')
parser.add_argument('--hierarchy_lvl', '-hl', default=1, type=int)
# for multi-channel support:
parser.add_argument('--stages_channels', nargs='+', type=str, default=[], help='List of stage:channel:params configurations')
parser.add_argument('--MC_encoder_type', choices=['TSSE', 'concat', 'hybrid'], default='TSSE')

default_parameters = {
    'enc': {
        "value":
            {'size': 2048, 'sparsity': 0.02}  #0.02
    },
    'predictor': {'sdrc_alpha': 0.1},
    'sp': {'boostStrength': 3.0,
           'localAreaDensity': 0.02,
           'potentialPct': 0.85,
           'synPermActiveInc': 0.04,
           'synPermConnected': 0.13999999999999999,
           'synPermInactiveDec': 0.006},
    'tm': {'activationThreshold': 8,
           'cellsPerColumn': 5,
           'initialPerm': 0.21,
           'maxSegmentsPerCell': 32,
           'maxSynapsesPerSegment': 256,
           'minThreshold': 3,
           'synPermConnected': 0.13999999999999999,
           'permanenceDec': 0.001,
           'cellNewConnectionMaxSegmentsGap': 0,
           'permanenceInc': 0.1},
    'anomaly': {'period': 1000},
}

def parse_stage_channel_config(config_str):
    """
    Parse a single stage:channel:params string into a tuple
    Example: 'P1:LIT101:window=5,sdr_size=1024->
            ('P1', 'LIT101', {'window': '5', 'sdr_size': '1024'})
    """
    stage, channel, params_str = config_str.split(':')
    params = {}
    if params_str:
        for param in params_str.split(','):
            key, value = param.split('=')
            params[key] = value
    return (stage, channel, params)


def main(args):
    print('running ...')

    # for multi-channel support
    configs = [parse_stage_channel_config(config) for config in args.stages_channels]
    is_multi_channel = len(configs) > 0

    
    file_prefix = swat_utils.get_file_prefix(args, args.channel_name)
    output_filepath = ''.join([args.output_file_path, file_prefix])
    input_filepath = ''.join([args.input_file_path, args.stage_name, '_data.csv'])
    meta_filepath = ''.join([args.input_file_path, args.stage_name, '_meta.csv'])

    if is_multi_channel:
        file_prefix = []
        for stage_name, channel_name, _ in configs:
            args.stage_name = stage_name
            file_prefix.append(swat_utils.get_file_prefix(args, channel_name))
        channel_parts = []
        for stage_name, channel_name, _ in configs:
            channel_parts.append(f"{stage_name}_{channel_name}")
        combined_name = "mixed_channel__" + "__".join(channel_parts)
        output_filepath = ''.join([args.output_file_path, combined_name])
        input_filepath = [''.join([args.input_file_path, stage_name, '_data.csv']) for stage_name,_,_ in configs]
        meta_filepath = [''.join([args.input_file_path, stage_name, '_meta.csv']) for stage_name,_,_ in configs]
        args.hierarchy_lvl = len(configs)
        args.encoding_type = "raw"
        args.encoding_duration_enabled = False
        args.stage_name = [stage_name for stage_name,_,_ in configs]
        args.channel_name = [channel_name for _,channel_name,_ in configs]
        for _, _, params in configs:
            for param_name, param_value in params.items():
                if not hasattr(args, param_name):  
                    print(f'Illegal param: {param_name}\nStopping Execution')
                    return
                if not isinstance(getattr(args, param_name), list):
                    setattr(args, param_name, [])  
                getattr(args, param_name).append(int(param_value))
        args.search_best_parameters = False

    runtime_config = {'verbose': args.verbose,
                      'CustomMinMax': args.limits_enabled,
                      'CustomMin': args.custom_min,
                      'CustomMax': args.custom_max,
                      'learn_during_training_only': args.learn_type == "train_only",
                      'freeze_configuration': args.freeze_type,
                      'stage': args.stage_name,
                      'input_path': input_filepath,
                      'output_path': output_filepath,
                      'meta_path': meta_filepath,
                      'var_name': args.channel_name,
                      'window': args.window,
                      'channel_type': args.channel_type,
                      'diff_enabled': args.diff_enabled,
                      'replay_buffer': args.replay_buffer,
                      'encoding_type': args.encoding_type,
                      'sum_window': args.sum_window,
                      'sum_threshold': args.sum_threshold,
                      'encoding_duration_value': args.encoding_duration_value,
                      'encoding_duration_enabled': args.encoding_duration_enabled,
                      'sampling_interval': args.sampling,
                      'hierarchy_enabled': args.hierarchy_enabled,
                      'hierarchy_lvl': args.hierarchy_lvl
                      }
    parameters = default_parameters  # default params of htm
    parameters['enc']['size'] = args.sdr_size
    parameters['enc']['sparsity'] = args.sdr_sparsity
    parameters['tm']['cellNewConnectionMaxSegmentsGap'] = args.connection_segments_gap
    parameters['runtime_config'] = runtime_config
    # override params of htm:
    if len(args.override_parameters) > 0:
        records_list = [item for item in args.override_parameters.split('/')]
        for record in records_list:
            param_list = [item for item in record.split(',')]
            if len(param_list) != 4:
                print(f"illegal param definition {param_list}")
                return
            else:
                print(f"override parameter: {param_list}")
                group_name = param_list[0]
                param_name = param_list[1]
                param_val = int(param_list[2])
                param_res = int(param_list[3])
                if param_res == 1:
                    parameters[group_name][param_name] = param_val
                else:
                    parameters[group_name][param_name] = float(param_val) / param_res

    config = parameters['runtime_config']
    if not is_multi_channel:
        stage = config['stage']  # like P1 or P2 etc.
        input_data = swat_utils.read_input(config['input_path'], config['meta_path'], args.sampling)  # dict
        assert stage.casefold() == input_data['stage'].casefold(), 'illegal input stage'
        features_info = input_data['features']  # channel with stats like min, max values
        V1PrmName = config['var_name']  # channel name, like P102
        v1_idx = features_info[V1PrmName]['idx']

        print('\nrun Stage1: min/max/mean/var of training data')
        print('==============================================')
        stage1_data = profiler_stage1(input_data, v1_idx)
        parameters['runtime_config']['stage1_data'] = stage1_data
        n_records = len(input_data['records'])
        hierarchy_enabled = parameters['runtime_config']['hierarchy_enabled']
        print(f"training points count: {input_data['training_count']}")
        print(f"total points count: {n_records}")

    else:  # multi-channel
        input_data = [swat_utils.read_input(config['input_path'][i], config['meta_path'][i], args.sampling) for i in range(len(config['input_path']))]  
        features_info = [input_data[i]['features'] for i in range(len(input_data))]
        n_records = [len(input_data[i]['records']) for i in range(len(input_data))]
        V1PrmName = [config['var_name'][i] for i in range(len(config['var_name']))]
        v1_idx = [features_info[i][V1PrmName[i]]['idx'] for i in range(len(V1PrmName))]
        
        print('\nrun Stage1: min/max/mean/var of training data')
        print('==============================================')
        stage1_data = []
        for i in range(len(input_data)):
            print()
            print(f"channel: {V1PrmName[i]}")
            stage1_data.append(profiler_stage1(input_data[i], v1_idx[i]))
            print(f"training points count: {input_data[i]['training_count']}")
            print(f"total points count: {n_records[i]}")
        print()
        parameters['runtime_config']['stage1_data'] = stage1_data
        hierarchy_enabled = parameters['runtime_config']['hierarchy_enabled']

    verbose = parameters['runtime_config']['verbose']

    # this is relevant only for analog channels (ctype = 0), floating point
    # note: not used in multi-channel
    if args.search_best_parameters and args.channel_type == 0:
        print('Stage 2: find best parameters')
        print('=============================')
        w_arr = [1, 3, 5, 8, 13, 21, 34]
        sdr_arr = [256, 512, 1024, 2048]
        training_count = input_data['training_count']
        parameters['runtime_config']['max_records_to_run'] = training_count
        sum_window = parameters['runtime_config']['sum_window']
        sum_threshold = parameters['runtime_config']['sum_threshold']
        parameters['runtime_config']['verbose'] = False
        parameters['runtime_config']['hierarchy_enabled'] = False
        best_window = w_arr[0]
        best_sdr = sdr_arr[0]
        min_score = 999999
        param_perms = [(x, y) for x in sdr_arr for y in w_arr]
        for i, (sdr, window) in enumerate(param_perms):
            print(f'\n-----[ sdr = {sdr}, window = {window} ]-----')
            parameters['enc']['size'] = sdr
            parameters['runtime_config']['window'] = window
            res = runner(input_data, parameters)
            dtest = res["data"]["Anomaly Score"][0:training_count]
            df = pandas.DataFrame(data=dtest)
            scores = df.iloc[0:training_count, 0].rolling(sum_window, min_periods=1, center=False).sum()
            thresholded_scores = swat_utils.anomaly_score(scores, sum_threshold)
            scores_found = swat_utils.count_continuous_ones(thresholded_scores[int(training_count * 0.2):])
            print(f'\nwindow = {window} found {scores_found} sum_scores > {sum_threshold} ')
            if (scores_found < min_score):
                min_score = scores_found
                best_window = window
                best_sdr = sdr
                if min_score == 0:
                    break

        print(f'best window = {best_window}, sdr = {best_sdr} with {min_score} sum_scores > {sum_threshold} ')
        if not args.window_tight:
            if best_window == 1:
                best_window = 3
            elif best_window == 3:
                best_window = 5
            elif best_window == 5:
                best_window = 8
            elif best_window == 8:
                best_window = 13
            elif best_window == 13:
                best_window = 21

        parameters['runtime_config']['window'] = best_window
        parameters['enc']['size'] = best_sdr

    # note: not used in multi-channel
    if parameters['runtime_config']['encoding_duration_enabled'] and parameters['runtime_config'][
        'encoding_duration_value'] == 0:
        print('Stage 3 - Find encodings max duration')
        print('===================================')
        training_count = input_data['training_count']
        parameters['runtime_config']['max_records_to_run'] = training_count
        parameters['runtime_config']['verbose'] = False
        res = profiler_stage3(input_data, parameters)
        print(f'\nmax_encoding_duration = {res["max_encoding_duration"]}')
        delay_hist = res["delay_hist"]
        print(f'\ndelay_hist = {delay_hist}')
        parameters['runtime_config']["encoding_duration_value"] = res["max_encoding_duration"]
        if delay_hist[-1] == 0.0 and delay_hist[-2] == 0.0 and delay_hist[-3] == 0.0:
            parameters['runtime_config']['delay_hist'] = delay_hist
        else:
            print(f'\n delay encoding disabled due to statistics')

    print('Final Stage')
    print('===========')
    parameters['runtime_config']['max_records_to_run'] = n_records
    parameters['runtime_config']['verbose'] = verbose
    parameters['runtime_config']['hierarchy_enabled'] = hierarchy_enabled

    if is_multi_channel:         
        input_data = combine_input_data(input_data, v1_idx)

    res = runner(input_data, parameters, is_multi_channel)
    save_results(res)


def runner(input_data, parameters, is_multi_channel=False):
    """
    Processes input data, encodes it, feeds it through HTM, and outputs anomaly scores
    """
    config = parameters['runtime_config']
    verbose = config['verbose']
    stage1_data = config['stage1_data']
    learn_during_training_only = config['learn_during_training_only']
    freeze_trained_network = config['freeze_configuration'] == "end_training"
    freeze_during_training = config['freeze_configuration'] == "during_training"
    output_filepath = config['output_path']
    if is_multi_channel:
        attack_labels = input_data['attack_labels']

    channel_type = config['channel_type']

    training_count = input_data['training_count']
    features_info = input_data['features']
    records = input_data['records']

    sdr_size = parameters["enc"]["size"]
    sdr_sparsity = parameters["enc"]["sparsity"]

    if args.MC_encoder_type == 'concat':
        # need to adjust sdr_sparsity for each channel so that they have the same number of active bits
        max_sdr_size = max(sdr_size)
        target_active_bits = int(max_sdr_size * 0.02)  # no. of active bits we want for all channels
        
        # adjusted sparsity for each channel
        sdr_sparsity = [target_active_bits / size for size in sdr_size]
        print(f'adjusted sdr_sparsity: {sdr_sparsity}')

        # Verify all channels will have same number of active bits
        # active_bits = [int(sdr_size[i] * sdr_sparsity[i]) for i in range(len(sdr_size))]
        # print(f'adjusted active_bits: {active_bits}')
        # return

    if is_multi_channel:
        if args.MC_encoder_type == 'TSSE':
            Num_sdr = max([s for s in sdr_size])  # max sdr size of all channels
        elif args.MC_encoder_type == 'concat':
            Num_sdr = sum(sdr_size)  # sum of sdr sizes of all channels
        else:  # TODO: hybrid
            pass
    else:
        Num_sdr = sdr_size  # single channel

    black_list = white_black_list.get_black_list()  # anomaly
    white_list = white_black_list.get_white_list()  # non-anomaly

    hierarchy_enabled = config['hierarchy_enabled']
    hierarchy_lvl = config["hierarchy_lvl"]  # in multi-channel, this is the number of channels
    hierarchy_current_lvl = 1
    hierarchy_rng = random.Random()
    hierarchy_rng.seed(10)
    permutation_enc = list(range(Num_sdr))  # [0,...,sdr_size-1]
    hierarchy_rng.shuffle(permutation_enc)  # shuffle([0,...,sdr_size-1])
    permutation_cdt = list()
    permutation_cdt.append(list(range(Num_sdr)))
    permutation_cdt.append(list(range(Num_sdr)))  # [[0,...,sdr_size-1],[0,...,sdr_size-1]]
    hierarchy_rng.shuffle(permutation_cdt[0])
    hierarchy_rng.shuffle(permutation_cdt[1])  # [shuffle([0,...,sdr_size-1]), shuffle([0,...,sdr_size-1])]
    hierarchy_sdr_buffer = collections.deque(maxlen=hierarchy_lvl)

    V1PrmName = config['var_name']  # channel name/s
    var_white_list = []
    var_black_list = []
    if not is_multi_channel:
        if V1PrmName in white_list.keys():
            var_white_list = white_list[V1PrmName]
            print(f'white list {var_white_list}')
        if V1PrmName in black_list.keys():
            var_black_list = black_list[V1PrmName]
            print(f'black list {var_black_list}')

    if is_multi_channel:
        v1_idx = [features_info[i][V1PrmName[i]]['idx'] for i in range(len(V1PrmName))]
    else:
        v1_idx = features_info[V1PrmName]['idx']

    # encoding set up
    if not is_multi_channel:    
        V1EncoderParams = ScalarEncoderParameters()
        V1EncoderParams.minimum, V1EncoderParams.maximum = max_min_values(config, features_info[V1PrmName], stage1_data)
    else:  # multi-channel
        V1EncoderParams = [ScalarEncoderParameters() for i in range(len(V1PrmName))]  # separate encoder for each channel
        for i in range(len(V1PrmName)):
            V1EncoderParams[i].minimum, V1EncoderParams[i].maximum = max_min_values(config, features_info[i][V1PrmName[i]], stage1_data[i])

    # TODO: something here needs to change for multi-channel
    active_bits = 0
    if config['channel_type'] == 0:  # continuous
        if not is_multi_channel:
            V1EncoderParams.size = sdr_size
            V1EncoderParams.sparsity = sdr_sparsity
            active_bits = int(sdr_size * sdr_sparsity)
        else:  # multi-channel
            active_bits = []
            for i in range(len(V1PrmName)):
                V1EncoderParams[i].size = sdr_size[i]
                if args.MC_encoder_type == 'TSSE':
                    V1EncoderParams[i].sparsity = sdr_sparsity
                    active_bits.append(int(sdr_size[i] * sdr_sparsity))
                elif args.MC_encoder_type == 'concat':
                    V1EncoderParams[i].sparsity = sdr_sparsity[i]
                    active_bits.append(int(sdr_size[i] * sdr_sparsity[i]))
                else:  # TODO: hybrid
                    pass
    else:  # scalar
        V1EncoderParams.category = 1
        active_bits = int(sdr_size / (V1EncoderParams.maximum - V1EncoderParams.minimum + 1))
        V1EncoderParams.activeBits = active_bits
        sdr_sparsity = float(V1EncoderParams.activeBits / sdr_size)

    if not is_multi_channel:
        V1Encoder = ScalarEncoder(V1EncoderParams)  # encoder
        print(f'active bits: {active_bits}')
        print(f'encoder min: {V1EncoderParams.minimum:.4}, max: {V1EncoderParams.maximum:.4}')
        print()
    else:  # multi-channel
        V1Encoder = [ScalarEncoder(V1EncoderParams[i]) for i in range(len(V1PrmName))]  # separate encoder for each channel
        for i in range(len(V1PrmName)):
            print(f'channel {V1PrmName[i]} encoder parameters:')
            print(f'active bits: {active_bits[i]}')
            print(f'encoder min: {V1EncoderParams[i].minimum:.4}, max: {V1EncoderParams[i].maximum:.4}')
            print()

    encoding_map_idx = 0
    if not is_multi_channel:
        V1EncodingSize = V1Encoder.size
        total_encoding_width = V1EncodingSize
        encoding_map = []
        default_encoding = list(range(total_encoding_width))
        encoding_map.append(default_encoding)
    else:  # multi-channel
        V1EncodingSize = [V1Encoder[i].size for i in range(len(V1PrmName))]
        total_encoding_width = Num_sdr

    # not used in multi-channel
    delay_encoding_enabled = False
    if 'delay_hist' in config.keys():
        delay_encoding_enabled = True
        delay_bins_list = [5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 17711, 28657]
        n_delay_bins_list = len(delay_bins_list)
        delay_hist = config['delay_hist']

        for idx, bin in enumerate(reversed(delay_hist)):
            if bin:
                delay_bins = [delay_bins_list[n_delay_bins_list - idx + 1],
                              delay_bins_list[n_delay_bins_list - idx + 2]]
                # delay_bins = [delay_bins_list[n_delay_bins_list - idx],delay_bins_list[n_delay_bins_list - idx + 1],delay_bins_list[n_delay_bins_list - idx + 2]]
                # delay_bins = [delay_bins_list[n_delay_bins_list - idx + 2]]
                break

        # delay_bins = [987]
        #delay_bins = [5,8,13,21,34,55,89,144,233]
        encoding_rng = random.Random()
        for idx, val in enumerate(delay_bins):
            random_encoding = default_encoding.copy()
            encoding_rng.seed(val)
            encoding_rng.shuffle(random_encoding)
            encoding_map.append(random_encoding)

        prev_delay_bin_idx = 0
        delay_bin_idx = 0
        delay_value = config["encoding_duration_value"]
        delay_encoding_width = swat_utils.get_delay_sdr_width(len(delay_bins) + 1)

        parameters['runtime_config']['delay_bins'] = delay_bins

    required_columns_for_prediction = [0, 1]  # TODO maybe change for multi-channel
    # required_columns_for_prediction = [0] * (delay_encoding_width+2)
    # required_columns_for_prediction[0] = swat_utils.get_delay_active_columns_num(delay_encoding_width)
    # required_columns_for_prediction[1] = active_bits+1
    # for i in range(delay_encoding_width):
    #     required_columns_for_prediction[2+i] = V1EncodingSize+i

    if not is_multi_channel:
        enc_info = Metrics([total_encoding_width], 999999999)
    else:  # multi-channel
        enc_info = Metrics([Num_sdr], 999999999)

    # Make the HTM and TemporalMemory & associated tools.
    activation_threshold = parameters["tm"]["activationThreshold"]
    
    # spParams = parameters["sp"]
    # sp = SpatialPooler(
    #     inputDimensions            = (encodingWidth,),
    #     columnDimensions           = (sdr_size,),
    #     potentialPct               = spParams["potentialPct"],
    #     potentialRadius            = encodingWidth,
    #     globalInhibition           = True,
    #     localAreaDensity           = spParams["localAreaDensity"],
    #     synPermInactiveDec         = spParams["synPermInactiveDec"],
    #     synPermActiveInc           = spParams["synPermActiveInc"],
    #     synPermConnected           = spParams["synPermConnected"],
    #     boostStrength              = spParams["boostStrength"],
    #     wrapAround                 = True
    # )
    # sp_info = Metrics( sp.getColumnDimensions(), 999999999 )
    total_active_bits = 0
    if args.MC_encoder_type == 'TSSE':
        total_active_bits = int(sdr_sparsity * Num_sdr)
    elif args.MC_encoder_type == 'concat':  
        for i in range(len(V1PrmName)):
            total_active_bits += sdr_sparsity[i] * sdr_size[i]
        total_active_bits = int(total_active_bits)
    else:  # TODO: hybrid
        pass
    
    tmParams = parameters["tm"]
    tm = TemporalMemory(
        columnDimensions=(total_encoding_width,),
        cellsPerColumn=tmParams["cellsPerColumn"],
        activationThreshold=activation_threshold,
        initialPermanence=tmParams["initialPerm"],
        connectedPermanence=tmParams["synPermConnected"],
        minThreshold=tmParams["minThreshold"],
        maxNewSynapseCount=total_active_bits,
        permanenceIncrement=tmParams["permanenceInc"],
        permanenceDecrement=tmParams["permanenceDec"],
        cellNewConnectionMaxSegmentsGap=tmParams["cellNewConnectionMaxSegmentsGap"],
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=tmParams["maxSegmentsPerCell"],
        maxSynapsesPerSegment=tmParams["maxSynapsesPerSegment"]
    )
    tm_info = Metrics([tm.numberOfCells()], 999999999)

    anomaly_history = AnomalyLikelihood(parameters["anomaly"]["period"])

    # Iterate through every datum in the dataset, record the inputs & outputs.

    N = len(records) if not is_multi_channel else len(records)//len(V1PrmName)
    inputs = [0.0] * N
    anomaly = [0.0] * N
    attack_label = [0] * (N - training_count)
    anomalyProb = [0] * N

    if not is_multi_channel:
        attack_label_idx = features_info["Attack"]['idx']
        v1_prev_init = True
        v1_prev = 0
        window_prev = 0
        prev_val1_encoding = 0
        test_count = 0
        window = config["window"]
    else:  # multi-channel
        attack_label_idx = [features_info[i]["Attack"]['idx'] for i in range(len(V1PrmName))]
        v1_prev_init = [True] * len(V1PrmName)
        v1_prev = [0] * len(V1PrmName)
        window_prev = [0] * len(V1PrmName)
        prev_val1_encoding = [0] * len(V1PrmName)
        test_count = 0
        window = config["window"]
    diff_enabled = config['diff_enabled']
    replay_buffer = config['replay_buffer']
    encoding_type = config['encoding_type']
    max_records_to_run = config['max_records_to_run']

    current_encoding_duration = 0

    if replay_buffer:
        sdr_rbuffer = collections.deque(maxlen=replay_buffer)

    if not is_multi_channel:
        if window > 1:
            val_buffer = collections.deque(maxlen=window)
    else:  # multi-channel
        val_buffer = [collections.deque(maxlen=window[i]) for i in range(len(V1PrmName))]

    tm.set_required_columns_for_prediction(required_columns_for_prediction)

    # ======================================================
    # ==================== main loop =======================
    # ======================================================
    test_input = [500, 510, 520, 510, 520, 510, 520, 510, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520]
    test_enabled = False
    # test_enabled = True
    # max_records_to_run = len(test_input)

    if verbose:
        import pprint
        pprint.pprint(parameters, indent=4)
        prm_output_filepath = ''.join([output_filepath, '_param.txt'])
        with open(prm_output_filepath, 'w') as f:
            pprint.pprint(parameters, indent=4, stream=f)
            pprint.pprint(f"training points count: {training_count}", indent=4, stream=f)
            pprint.pprint(f"total points count: {len(records)}", indent=4, stream=f)
            pprint.pprint(features_info, indent=4, stream=f)
        prm_output_filepath_json = ''.join([output_filepath, '_param.json'])
        with open(prm_output_filepath_json, 'w') as fj:
            fj.write(json.dumps(parameters))

    init2 = True
    curr_channel = 0  # used in multi-channel to select the correct channel

    # max_records_to_run = 100_000  # TODO remove
    for count, record in enumerate(records):
        if count % 100_000 == 0:
            print(f'count: {count}')    

        if count == max_records_to_run:
            break
        
        if is_multi_channel:
            curr_channel = count % len(V1PrmName)

        # get value from record specific to the channel
        if test_enabled and count < len(test_input):
            # test is from above
            record_val = float(test_input[count])
        else:
            if not is_multi_channel:
                record_val = float(record[v1_idx])
            else:
                record_val = float(record)
        
        # truncate to min/max TODO are enc.min/max from the train set?
        if not diff_enabled:
            if not is_multi_channel:
                record_val = keep_limits(record_val, V1EncoderParams.minimum, V1EncoderParams.maximum)
            else:
                record_val = keep_limits(record_val, V1EncoderParams[curr_channel].minimum, V1EncoderParams[curr_channel].maximum)
        
        # sliding window of values
        if not is_multi_channel:
            if window > 1:
                val_buffer.append(record_val)
                n = 0.0
                s = 0.0
                for v in val_buffer:
                    s += v
                    n += 1

                window_val = s / n  # avg value in val_buffer
            else:
                window_val = record_val
        else:  # multi-channel
            val_buffer[curr_channel].append(record_val)
            n = 0.0
            s = 0.0
            for v in val_buffer[curr_channel]:
                s += v
                n += 1
            window_val = s / n  # avg value in val_buffer

        # diff values
        if diff_enabled:  # derivative of continuous channel
            if count == 0:
                window_prev = window_val
                continue
            v1_val = window_val - window_prev
            v1_val = keep_limits(v1_val, V1EncoderParams.minimum, V1EncoderParams.maximum)
        else:  # continuous channel value
            v1_val = window_val
        window_prev = window_val

        if v1_prev_init:
            v1_prev_init = False
            v1_prev = v1_val

        if not is_multi_channel:
            inputs[count] = v1_val  
        # TODO maybe needed for multi-channel

        # save anomaly labels for test data
        if not is_multi_channel:
            if count >= training_count:
                attack_label[count - training_count] = int(record[attack_label_idx])
                test_count += 1
        else:  # multi-channel
            if count//len(V1PrmName) >= training_count:
                # assuming all channels should have the same attack labels for a given time point
                if curr_channel == 0:  # Only save on first channel's iteration
                    attack_label[count//len(V1PrmName) - training_count] = attack_labels[count//len(V1PrmName)]
                    test_count += 1

        # call the encoders to create bit representations for each value. these are SDR objects.

        # discrete channel values or dAk or Ak with hier buffer size of 1
        if channel_type == 1 or hierarchy_enabled == False or hierarchy_lvl == 1:
            default_encoding = V1Encoder.encode(v1_val)
        # Ak with hier buffer size > 1 or multi-channel
        else:
            if not is_multi_channel:    
                hierarchy_sdr_buffer.append(swat_utils.SDR2blist(V1Encoder.encode(v1_val)))
            else:  # multi-channel
                if args.MC_encoder_type == 'TSSE':
                    hierarchy_sdr_buffer.append(swat_utils.pad_binary_list(swat_utils.SDR2blist(V1Encoder[curr_channel].encode(v1_val)), Num_sdr))
                elif args.MC_encoder_type == 'concat':
                    hierarchy_sdr_buffer.append(swat_utils.SDR2blist(V1Encoder[curr_channel].encode(v1_val)))
                else:  # TODO: hybrid
                    pass

            if hierarchy_current_lvl == hierarchy_lvl:
                if args.MC_encoder_type == 'TSSE':
                    sdr_encoded_bin = swat_utils.encode_sequence_fast(list(hierarchy_sdr_buffer), permutation_enc)
                    sdr_cdt_bin, N0, N1 = swat_utils.stable_cdt_fast(sdr_encoded_bin, sdr_sparsity, permutation_cdt)
                    default_encoding = swat_utils.blist2SDR(sdr_cdt_bin)
                elif args.MC_encoder_type == 'concat':
                    # default_encoding = []
                    # for i in range(len(V1PrmName)):
                    #     # concat all the SDRs
                    #     default_encoding.extend(hierarchy_sdr_buffer[i])
                    # default_encoding = swat_utils.blist2SDR(default_encoding)
                    default_encoding_bin = np.concatenate(hierarchy_sdr_buffer)
                    default_encoding = swat_utils.blist2SDR_fast(default_encoding_bin)
                else:  # TODO: hybrid
                    pass

                if is_multi_channel:
                    hierarchy_current_lvl = 1 
                    hierarchy_sdr_buffer.clear()
            else:
                hierarchy_current_lvl = hierarchy_current_lvl + 1
                continue

        # encoding_map_idx 0 is the default 1:1 encoding
        if encoding_map_idx:
            val1_encoding = SDR(total_encoding_width)
            val1_encoding.sparse = [encoding_map[encoding_map_idx][i] for i in default_encoding.sparse]
        else:
            val1_encoding = default_encoding

        # add SDR to replay buffer
        if replay_buffer:
            sdr_rbuffer.append(val1_encoding)

        training_period = count < training_count
        learn = True
        if learn_during_training_only:
            learn = count < training_count

        permanent = False
        if freeze_during_training:
            permanent = training_period

        if freeze_trained_network and count == training_count:
            tm.make_current_network_permanent()
            print('training done, freeze network..')

        val_is_white = False
        val_is_black = False

        if not training_period:
            for vv in var_white_list:
                if v1_prev == vv[0] and v1_val == vv[1]:
                    val_is_white = True
                    break

            for vv in var_black_list:
                if v1_prev == vv[0] and v1_val == vv[1]:
                    val_is_black = True
                    break
        
        run_tm = False
        if encoding_type == 'raw':
            run_tm = True
        if encoding_type == 'diff':
            if init2:
                run_tm = True
                init2 = False
                prev_default_encoding_delay = default_encoding
                prev_val1_encoding = val1_encoding

            if prev_val1_encoding != val1_encoding:
                run_tm = True

            if delay_encoding_enabled:
                delay_bin_idx = swat_utils.get_delay_bin_idx(delay_bins, current_encoding_duration)
                if delay_bin_idx != prev_delay_bin_idx:
                    encoding_map_idx = delay_bin_idx
                    prev_delay_bin_idx = delay_bin_idx

                if count < 2 or (default_encoding.getOverlap(prev_default_encoding_delay) < active_bits * 0.7):
                    prev_default_encoding_delay = default_encoding
                    current_encoding_duration = 0
                else:
                    current_encoding_duration += 1

        if run_tm:
            encoding = val1_encoding

            enc_info.addData(encoding)
            if val_is_black:
                tm.compute(encoding, learn=False, permanent=False)
            else:
                tm.compute(encoding, learn=learn, permanent=permanent)

            tm_active_cells = tm.getActiveCells()
            tm_info.addData(tm_active_cells.flatten())
            if val_is_black:
                print(f'black list: prev_val = {v1_prev}, curr_val = {v1_val}')
                anomaly[count] = 1.0
                anomalyProb[count] = 1.0
            elif val_is_white:
                print(f'white list: prev_val = {v1_prev}, curr_val = {v1_val}')
                anomaly[count] = 0.0
                anomalyProb[count] = 0.0
            else:
                # anomaly[count] = tm.anomaly  
                predicted = tm.getPredictedColumns()
                # print(f'predicted: {predicted}')
                if not is_multi_channel:
                    anomaly[count] = swat_utils.computeAnomalyScore(encoding, predicted)
                    # assert math.fabs(score - tm.anomaly) < 0.0000001, "anomaly calculation wrong"
                    anomalyProb[count] = anomaly_history.compute(anomaly[count])
                else:  # multi-channel: only gets here after TSSE / hierarchy_buffer is full
                    anomaly[count // len(V1PrmName)] = swat_utils.computeAnomalyScore(encoding, predicted)
                    anomalyProb[count // len(V1PrmName)] = anomaly_history.compute(anomaly[count // len(V1PrmName)])

            if replay_buffer and tm.anomaly:
                for replay_enc in sdr_rbuffer:
                    tm.compute(replay_enc, learn=False, permanent=False)
        else:
            # no anomaly occurred
            anomaly[count] = 0.0
            anomalyProb[count] = 0.0

        prev_val1_encoding = val1_encoding
        v1_prev = v1_val

        if not is_multi_channel:
            print_progress(count)

    if verbose:
        # Print information & statistics about the state of the HTM.
        print("Encoded Input", enc_info)
        print("")
        print("Temporal Memory Cells", tm_info)
        print(str(tm))
        print("")
        # Show info about the anomaly (mean & std)
        print("Anomaly Mean", np.mean(anomaly))
        print("Anomaly Std ", np.std(anomaly))

        prm_output_filepath = ''.join([output_filepath, '_stats.txt'])
        with open(prm_output_filepath, 'w') as f:
            pprint.pprint("Encoded Input", indent=4, stream=f)
            pprint.pprint(str(enc_info), indent=4, stream=f)
            pprint.pprint("", indent=4, stream=f)
            pprint.pprint("Temporal Memory", indent=4, stream=f)
            pprint.pprint(str(tm_info), indent=4, stream=f)
            pprint.pprint("", indent=4, stream=f)
            pprint.pprint(str(tm), indent=4, stream=f)
            pprint.pprint("", indent=4, stream=f)
            pprint.pprint(f"Anomaly Mean: {np.mean(anomaly)}", indent=4, stream=f)
            pprint.pprint(f"Anomaly STD: {np.std(anomaly)}", indent=4, stream=f)

    # placeholder..
    pred1 = anomaly
    pred5 = anomaly
    # end placeholder

    data = {"Input": inputs, "1 Step Prediction": pred1, "5 Step Prediction": pred5,
            "Anomaly Score": anomaly, "Anomaly Likelihood": anomalyProb}
    result = {"data": data, "output_filepath": output_filepath, "attack_label": attack_label, "test_count": test_count}
    return result


def save_results(result):
    df = pandas.DataFrame(result["data"])
    htm_output_filepath = ''.join([result["output_filepath"], '_res.csv'])
    print(f'htm_output_filepath: {htm_output_filepath}')
    df.to_csv(htm_output_filepath, sep=',', index=False)
    attack_label_output_filepath = ''.join([result["output_filepath"], '_attack.real'])
    print(f'attack_label_output_filepath: {attack_label_output_filepath}')
    swat_utils.save_list(result["attack_label"], attack_label_output_filepath)
    print(f'test_count: {result["test_count"]}, len: {len(result["attack_label"])}')


def profiler_stage1(input_data, var_idx):
    training_count = input_data['training_count']
    N_profile = training_count // 10

    records = input_data['records']

    min_val = 0
    max_val = 0
    min_diff = 0
    max_diff = 0
    prev_val = 0
    n = 0
    mean_val = 0
    var_val = 0
    mean_diff = 0
    var_diff = 0

    for count, record in enumerate(records):
        val = float(record[var_idx])
        if count == 0:
            min_val = val
            max_val = val
            prev_val = val
            continue

        min_val = min(min_val, val)
        max_val = max(max_val, val)

        diff_val = val - prev_val
        if count == 1:
            min_diff = diff_val
            max_diff = diff_val
            continue

        min_diff = min(min_diff, diff_val)
        max_diff = max(max_diff, diff_val)

        n = n + 1
        mean_val += val
        mean_diff += diff_val

        print_progress(count)
        if count == N_profile:
            break

    mean_val = mean_val / n
    mean_diff = mean_diff / (n - 1)

    print('\nStage1 Profiler: calc variance')
    for count, record in enumerate(records):
        val = float(record[var_idx])
        var_val += (val - mean_val) ** 2

        if count == 0:
            prev_val = val
            continue

        diff_val = val - prev_val
        var_diff += (diff_val - mean_diff) ** 2

        print_progress(count)

        if count == N_profile:
            break

    var_val = var_val / (n - 1)
    var_diff = var_diff / (n - 2)

    print(f'\nStage1 Profiler Done Using {N_profile} Samples')
    print(f'Min Value: {min_val:.4}, Max Value: {max_val:.4}, Mean Value: {mean_val:.4}, Var Value: {var_val:.4}')
    print(f'Min Diff: {min_diff:.4}, Max Diff: {max_diff:.4}, Mean Diff {mean_diff:.4}, Var Diff: {var_diff:.4}')

    stage1_data = {'value': {'min': min_val, 'max': max_val, 'mean': mean_val, 'var': var_val},
                   'diff': {'min': min_diff, 'max': max_diff, 'mean': mean_diff, 'var': var_diff}}

    return stage1_data


def profiler_stage3(input_data, parameters):
    config = parameters['runtime_config']
    features_info = input_data['features']
    records = input_data['records']
    stage1_data = config['stage1_data']
    sdr_size = parameters["enc"]["size"]
    sdr_sparsity = parameters["enc"]["sparsity"]

    V1PrmName = config['var_name']
    v1_idx = features_info[V1PrmName]['idx']

    V1EncoderParams = ScalarEncoderParameters()
    V1EncoderParams.minimum, V1EncoderParams.maximum = max_min_values(config, features_info[V1PrmName], stage1_data)

    if config['channel_type'] == 0:
        V1EncoderParams.size = sdr_size
        V1EncoderParams.sparsity = sdr_sparsity
        active_bits = int(sdr_size * sdr_sparsity)
    else:
        V1EncoderParams.category = 1

        active_bits = int(sdr_size / (V1EncoderParams.maximum - V1EncoderParams.minimum + 1))
        V1EncoderParams.activeBits = active_bits

    V1Encoder = ScalarEncoder(V1EncoderParams)

    print(f'active bits: {active_bits}')
    print(f'encoder min: {V1EncoderParams.minimum:.4}, max: {V1EncoderParams.maximum:.4}')

    delay_bins = [5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 17711, 28657]
    delay_hist_count = 0
    prev_delay_bin_idx = 0
    delay_hist = [0] * (len(delay_bins) + 1)
    delay_hist_percent = [0] * (len(delay_bins) + 1)

    v1_prev_init = True
    window_prev = 0
    window = config["window"]
    diff_enabled = config['diff_enabled']
    max_records_to_run = config['max_records_to_run']

    max_encoding_duration = 0
    current_encoding_duration = 0

    if window > 1:
        val_buffer = collections.deque(maxlen=window)

    # ======================================================
    # ==================== main loop =======================
    # ======================================================

    for count, record in enumerate(records):
        if count == max_records_to_run:
            break

        record_val = float(record[v1_idx])

        if not diff_enabled:
            record_val = keep_limits(record_val, V1EncoderParams.minimum, V1EncoderParams.maximum)

        # sliding window
        if window > 1:
            val_buffer.append(record_val)
            n = 0.0
            s = 0.0
            for v in val_buffer:
                s += v
                n += 1

            window_val = s / n
        else:
            window_val = record_val

        # diff values
        if diff_enabled:
            if count == 0:
                window_prev = window_val
                continue
            v1_val = window_val - window_prev
            v1_val = keep_limits(v1_val, V1EncoderParams.minimum, V1EncoderParams.maximum)
        else:
            v1_val = window_val

        window_prev = window_val

        if v1_prev_init:
            v1_prev_init = False

        default_encoding = V1Encoder.encode(v1_val)

        if count < 2:
            prev_default_encoding_delay = default_encoding

        delay_bin_idx = swat_utils.get_delay_bin_idx(delay_bins, current_encoding_duration)
        if delay_bin_idx != prev_delay_bin_idx:
            prev_delay_bin_idx = delay_bin_idx

        if count < 2 or (default_encoding.getOverlap(prev_default_encoding_delay) < active_bits * 0.7):
            max_encoding_duration = max(current_encoding_duration, max_encoding_duration)
            delay_hist[delay_bin_idx] += 1
            delay_hist_count += 1

            prev_default_encoding_delay = default_encoding
            current_encoding_duration = 0
        else:
            current_encoding_duration += 1

    if delay_hist_count:
        for i, v in enumerate(delay_hist):
            delay_hist_percent[i] = delay_hist[i] * 100.0 / delay_hist_count

    result = {'max_encoding_duration': max_encoding_duration, 'delay_hist': delay_hist_percent}
    return result


def print_progress(count):
    if count > 1 and count % 100000 == 0:
        print(f"{count}")
    if count < 5 or count % 10000 == 0:
        print(".", end=" ")


def max_min_values(config, var_info, stage1_data):
    min_val = 0
    max_val = 0
    if config['CustomMinMax'] is True:
        print("Custom MinMax")
        min_val = float(config['CustomMin'])
        max_val = float(config['CustomMax'])
    else:
        if config['channel_type'] == 0:
            if config['diff_enabled']:
                addon = 0.05 * (stage1_data['diff']['max'] - stage1_data['diff']['min'])
                min_val = stage1_data['diff']['min'] - addon
                max_val = stage1_data['diff']['max'] + addon
            else:
                addon = 0.05 * (stage1_data['value']['max'] - stage1_data['value']['min'])
                min_val = stage1_data['value']['min'] - addon
                max_val = stage1_data['value']['max'] + addon
        else:
            min_val = var_info['min']
            max_val = var_info['max']

    return min_val, max_val


def keep_limits(val, min_val, max_val):
    # force val to be in (min_val, max_val) if outside this range
    if val < min_val:
        val = min_val + 0.000001

    if val > max_val:
        val = max_val - 0.000001

    return val


def combine_input_data(channel_data_list, indexes):
    """
    Combines input data from multiple channels into a single input data.
    Args:
        channel_data_list: List of input data dictionaries, one per channel
        indexes: List of indexes for each channel
    Returns:
        combined_input_data: dict containing combined data
    """
    combined_records = []
    attack_labels = []  # attack label is the same for all channels
    attack_label_idx = channel_data_list[0]["features"]["Attack"]["idx"]
    for i in range(len(channel_data_list[0]['records'])):  # number of records
        attack_labels.append(channel_data_list[0]["records"][i][attack_label_idx])
        for j in range(len(channel_data_list)):  # number of channels
            combined_records.append(channel_data_list[j]['records'][i][indexes[j]])
    
    combined_meta = [channel_data_list[i]['meta'] for i in range(len(channel_data_list))]
    combined_features = [channel_data_list[i]['features'] for i in range(len(channel_data_list))]
    combined_stage = [channel_data_list[i]['stage'] for i in range(len(channel_data_list))]
    combined_training_count = [channel_data_list[i]['training_count'] for i in range(len(channel_data_list))]

    combined_input_data = {
        'stage': combined_stage,
        'features': combined_features,
        'records': combined_records,
        'training_count': combined_training_count[0],
        'meta': combined_meta,
        'attack_labels': attack_labels
    }
    return combined_input_data

if __name__ == "__main__":
    # sys.argv = ['swat_htm.py',
    #             '--stage_name', 'P1',
    #             '--channel_name', 'LIT101',
    #             '--freeze_type', 'off',
    #             '--learn_type', 'always',
    #             '--verbose',
    #             '-ctype','0',
    #             '-sbp','-w','5','-size','1024','-ed_val','1']

    args = parser.parse_args()
    print(args)
    main(args)
