import csv
from htm.bindings.sdr import SDR
import random
import numpy as np

def read_input(input_path, meta_path, sampling_interval):
    """
    reads the preprocessed input data and metadata for running HTM on SWaT dataset.
    :param input_path: path to preprocessed input data: discrete and/or analog.
    :param meta_path: path to metadata file: includes things like stage name, num of records, num of features, ...
    :param sampling_interval: non-negative integer, store the record every sampling_interval.
    :return: dict containing metadata, preprocessed input data (records), stage no., num of records, num of features
    """

    meta = []
    records = []
    sampling_count = 0

    # open input data, and store the records every sampling_interval
    with open(input_path, "r") as fin:
        reader = csv.reader(fin)
        for record in reader:
            if sampling_count == 0:
                records.append(record)
            sampling_count += 1
            if sampling_count == sampling_interval:
                sampling_count = 0

    # open and parse metadata
    with open(meta_path, "r") as fin:
        reader = csv.reader(fin)
        for record in reader:
            meta.extend(record)

    features_info = dict()
    for idx in range(int(meta[2])):
        pos = 3 + 3 * idx
        features_info.update({str(meta[pos]): {'idx': idx, 'min': int(meta[pos + 1]), 'max': int(meta[pos + 2])}})

    input_data = {'meta': meta,
                  'records': records,
                  'stage': meta[0],
                  'training_count': int(meta[1]) // sampling_interval,
                  'features': features_info}

    return input_data


def get_file_prefix(args, channel_name):
    file_prefix = args.stage_name
    file_prefix += '_'
    file_prefix += channel_name
    file_prefix += "_learn_"
    file_prefix += args.learn_type
    file_prefix += "_freeze_"
    file_prefix += args.freeze_type
    if args.prefix != "":
        file_prefix += '_'
        file_prefix += args.prefix

    return file_prefix


def save_list(data, output_path):
    with open(output_path, 'w') as fp:
        for item in data:
            fp.write("%s\n" % item)

    return


def anomaly_score(data, threshold):
    """
    calculates anomaly scores
    :param data:
    :param threshold:
    :return: list of len of data, filled with 1 if data[i] >= threshold, o.w. 0
    """
    res = [None] * len(data)
    for idx, item in enumerate(data):
        res[idx] = 1 if item >= threshold else 0

    return res


def count_continuous_ones(data):
    """
    counts the number of continuous sequences of ones in the data.
    :param data: list
    :return: number of continuous ones
    """
    prev_zero = True
    found = 0
    for d in data:
        if d:
            if prev_zero:
                prev_zero = False
                found = found + 1
        else:
            prev_zero = True

    return found


def calc_anomaly_stats(scores, labels, grace_time=0):
    """
    calculates anomaly statistics, like TP, FP, TN, FN.
    True Positives (TP): Correctly identified anomalies.
    False Positives (FP): Incorrectly identified anomalies.
    False Negatives (FN): Missed anomalies.
    :param scores: threshold scores (1s and 0s)
    :param labels: ground truth
    :param grace_time: length of time to allow the system to restabilize
    :return: a dict with anomaly stats.
    """
    l_count = 0
    s_false_count = 0
    TP_detected_labels = []
    TP_detection_delay = []
    l_start_time = 0
    N = len(labels)
    FP_arr = [0] * N
    FP_start_idx = 0
    stats = {}
    stats['TP'] = 0
    stats['FP'] = 0
    stats['FN'] = 0
    stats['PR'] = 0.0
    stats['RE'] = 0.0
    stats['F1'] = 0.0
    stats['detected_labels'] = []
    stats['detection_delay'] = []
    stats['fp_array'] = []
    stats['LabelsCount'] = 0
    label_grace_time = grace_time // 10
    last_label_detected = False

    if len(scores) != N:
        print(f'Error, labels{N} and anomaly{len(scores)} vectors length is different')
        return stats

    l_prev = False
    s_prev = False
    l_now = False
    s_now = False

    l_marked = False
    in_label = False
    s_marked = False
    start = True
    for idx, (score, label) in enumerate(zip(scores, labels)):
        #skip score tail from training..
        if start and label == 0 and score == 1:
            continue
        start = False

        l_prev = l_now
        l_now = True if label == 1 else False
        s_prev = s_now
        s_now = True if score == 1 else False

        if s_now and s_prev == False:
            FP_start_idx = idx

        if l_now and l_prev == False:
            l_count = l_count + 1
            in_label = True
            last_label_detected = False
            l_start_time = idx

        if l_now == False:
            in_label = False
            l_marked = False

        if in_label and l_marked == False and s_now:
            TP_detected_labels.append(l_count)
            TP_detection_delay.append(idx - l_start_time)
            last_label_detected = True
            l_marked = True

        if s_now and l_now:
            s_marked = True

        if (s_prev and s_now == False) or (s_now and idx == N - 1):
            if s_marked == False:
                if not last_label_detected:
                    max_hist = min(FP_start_idx, label_grace_time)
                    for i in range(max_hist):
                        if labels[FP_start_idx - i] == 1:
                            TP_detected_labels.append(l_count)
                            TP_detection_delay.append(idx - l_start_time)
                            s_marked = True
                            break

            if s_marked == False:
                max_hist = min(FP_start_idx, grace_time)
                for i in range(max_hist):
                    if FP_arr[FP_start_idx - i] == 1 or labels[FP_start_idx - i] == 1:
                        s_marked = True
                        break

            if s_marked == False:
                s_false_count = s_false_count + 1
                FP_arr[FP_start_idx:idx] = [1] * (idx - FP_start_idx)
                if s_now and idx == N - 1:
                    FP_arr[-1] = 1
            s_marked = False

    TP = len(TP_detected_labels)
    FN = l_count - TP
    FP = s_false_count
    PR = precision(TP, FP)
    RE = recall(TP, FN)
    # print(f'TP_detected_labels = {TP_detected_labels}')
    stats = {}
    stats['TP'] = TP
    stats['FP'] = FP
    stats['FN'] = FN
    stats['PR'] = PR
    stats['RE'] = RE
    stats['F1'] = F1(PR, RE)
    stats['detected_labels'] = stage_id_to_global_id(0, TP_detected_labels)
    stats['detection_delay'] = TP_detection_delay
    assert len(TP_detection_delay) == len(stats['detected_labels']), 'TP_detection_delay len error'
    stats['fp_array'] = FP_arr
    stats['LabelsCount'] = l_count

    return stats


def precision(TP, FP):
    return TP / (TP + FP) if TP + FP != 0 else 0.0


def recall(TP, FN):
    return TP / (TP + FN) if TP + FN != 0 else 0.0


def F1(PR, RE):
    return 2 * PR * RE / (PR + RE) if PR + RE != 0 else 0.0


def test_calc_anomaly_stats():
    #1
    l = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    idx = 0
    idx = idx + 1
    stats = calc_anomaly_stats(s, l)
    print(f'{idx}: {stats}')

    # 2
    l = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    s = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    idx = idx + 1
    stats = calc_anomaly_stats(s, l)
    print(f'{idx}: {stats}')

    #3
    l = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    s = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    idx = idx + 1
    stats = calc_anomaly_stats(s, l)
    print(f'{idx}: {stats}')

    #4
    l = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    s = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    idx = idx + 1
    stats = calc_anomaly_stats(s, l)
    print(f'{idx}: {stats}')

    #5
    l = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    s = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    idx = idx + 1
    stats = calc_anomaly_stats(s, l)
    print(f'{idx}: {stats}')

    #6
    l = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    s = [0, 0, 0, 1, 0, 1, 0, 1, 1, 0]
    idx = idx + 1
    stats = calc_anomaly_stats(s, l)
    print(f'{idx}: {stats}')

    #7
    l = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
    s = [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    idx = idx + 1
    stats = calc_anomaly_stats(s, l)
    print(f'{idx}: {stats}')

    #8
    l = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
    s = [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1]
    idx = idx + 1
    stats = calc_anomaly_stats(s, l)
    print(f'{idx}: {stats}')

    #9
    l = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    idx = idx + 1
    stats = calc_anomaly_stats(s, l)
    print(f'{idx}: {stats}')

    #10
    l = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    s = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    idx = idx + 1
    stats = calc_anomaly_stats(s, l)
    print(f'{idx}: {stats}')

    #11
    l = [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]
    s = [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
    idx = idx + 1
    stats = calc_anomaly_stats(s, l)
    print(f'{idx}: {stats}')

    #12
    l = [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1]
    s = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    idx = idx + 1
    stats = calc_anomaly_stats(s, l)
    print(f'{idx}: {stats}')


def test_count_continuous_ones():
    a = [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0]
    val = count_continuous_ones(a)
    print(f'found {val} continuous ones')


def computeAnomalyScore(active, predicted):
    """
    Calculates a number from 0 to 1, indicated how much predicted matches active.
    The higher the number, the less predicted matches active.
    :return: 0 if predicted matches active perfectly (no anomaly). 1 if predicted contains none of the bits in active (high anomaly).
    """
    if active.getSum() == 0:
        return 0.0  # no anomaly

    both = SDR(active.dimensions)
    both.intersection(active, predicted)  # bitwise and

    score = (active.getSum() - both.getSum()) / active.getSum()

    return score


def stage_id_to_global_id(stage_id, stage_id_list):
    # id:0 is all labels, id:1..6 is P1..P6
    stage_ids_map = [
        [1, 2, 3, 6, 7, 8, 10, 11, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
         39, 40, 41],
        [1, 2, 3, 21, 26, 27, 28, 30, 33, 34, 35, 36],
        [2, 6, 24, 26, 27, 28, 30],
        [7, 8, 16, 17, 23, 26, 27, 28, 32, 41],
        [8, 10, 11, 17, 22, 23, 25, 27, 28, 31, 37, 38, 39, 40],
        [10, 11, 19, 20, 22, 27, 28, 37, 38, 39, 40],
        [8, 23, 28]
    ]

    assert stage_id >= 0 and stage_id <= 6, 'illegal stage id'

    res = [stage_ids_map[stage_id][idx - 1] for idx in stage_id_list]
    return res


def get_delay_sdr_width(bin_size: int):
    assert bin_size >= 1 and bin_size <= 20, 'bin size > 20'
    if bin_size == 1:
        return 1
    if bin_size == 2:
        return 2
    if bin_size == 3:
        return 3
    if bin_size >= 4 and bin_size <= 6:
        return 4
    if bin_size >= 7 and bin_size <= 10:
        return 5
    if bin_size >= 11 and bin_size <= 20:
        return 6


def get_delay_bin_idx(bins, value):
    """
    examples:
    let bins = [5, 10, 15]
    if value = 7 then ret 1
    if value = 12 then ret 2
    if value = 20 then ret 3
    """
    for idx, bin_val in enumerate(bins):
        if value < bin_val:
            return idx

    return len(bins)


def get_delay_active_columns_num(sdr_len):
    assert sdr_len >= 0 and sdr_len <= 6, 'illegal sdr_len'
    val = [0, 1, 1, 2, 2, 2, 3]
    return val[sdr_len]


def get_delay_sdr(state_idx, sdr_len):
    s = [[[1]],  #1
         [[1, 0],  #2
          [0, 1]],
         [[1, 1, 0],  #3
          [1, 0, 1],
          [0, 1, 1]],
         [[1, 1, 0, 0],  #4
          [1, 0, 1, 0],
          [1, 0, 0, 1],
          [0, 1, 1, 0],
          [0, 1, 0, 1],
          [0, 0, 1, 1]],
         [[1, 1, 0, 0, 0],  #5
          [1, 0, 1, 0, 0],
          [1, 0, 0, 1, 0],
          [1, 0, 0, 0, 1],
          [0, 1, 1, 0, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 0, 0, 1],
          [0, 0, 1, 1, 0],
          [0, 0, 1, 0, 1],
          [0, 0, 0, 1, 1]],
         [[1, 1, 1, 0, 0, 0],  #6
          [1, 1, 0, 1, 0, 0],
          [1, 1, 0, 0, 1, 0],
          [1, 1, 0, 0, 0, 1],
          [1, 0, 1, 1, 0, 0],
          [1, 0, 1, 0, 1, 0],
          [1, 0, 1, 0, 0, 1],
          [1, 0, 0, 1, 1, 0],
          [1, 0, 0, 1, 0, 1],
          [1, 0, 0, 0, 1, 1],
          [0, 1, 1, 1, 0, 0],
          [0, 1, 1, 0, 1, 0],
          [0, 1, 1, 0, 0, 1],
          [0, 1, 0, 1, 1, 0],
          [0, 1, 0, 1, 0, 1],
          [0, 1, 0, 0, 1, 1],
          [0, 0, 1, 1, 1, 0],
          [0, 0, 1, 1, 0, 1],
          [0, 0, 1, 0, 1, 1],
          [0, 0, 0, 1, 1, 1]]]

    assert sdr_len <= 6, 'get_state_sdr: sdr_len >  6'
    assert state_idx < len(s[sdr_len - 1]), 'get_state_sdr: idx > max len'

    return s[sdr_len - 1][state_idx]


def test_get_state_sdr():
    s = get_delay_sdr(0, 2)
    assert s == [1, 0], 'get_state_sdr(0,2)'

    s = get_delay_sdr(1, 3)
    assert s == [1, 0, 1], ' get_state_sdr(1,3)'

    s = get_delay_sdr(19, 6)
    assert s == [0, 0, 0, 1, 1, 1], ' get_state_sdr(19,6)'

    print('test_get_state_sdr test done')


def test_stage_id_to_global_id():
    ids = stage_id_to_global_id(1, [1, 4, 11])
    assert ids == [1, 21, 36], 'mapping error'


def and_blist(x, y):
    return [a and b for a, b in zip(x, y)]


def or_blist(x, y):
    return [a or b for a, b in zip(x, y)]


def not_blist(x):
    return [not a for a in x]


def list2blist(x):
    return [a != 0 for a in x]


def blist2list(x):
    return [int(a) for a in x]


def SDR2blist(x):
    val = [False] * x.size
    for i in x.sparse:
        val[i] = True

    return val


def blist2SDR(x):
    res = SDR(len(x))
    res.sparse = [i for i, val in enumerate(x) if x[i]]

    return res

def blist2SDR_fast(x):
    res = SDR(len(x))
    if isinstance(x, np.ndarray):
        res.sparse = np.nonzero(x)[0].tolist()
    else:
        res.sparse = np.where(x)[0].tolist()
    return res


def stable_cdt(SDRT, target_sparsity, permutation, alpha=1.1):
    """
    step 3 of TSSE. makes SDRT an SDR with target_sparsity.
    :param SDRT: an SDR
    :param target_sparsity:
    :param permutation:
    :param alpha:
    :return: a single SDR with an acceptable sparsity <= target_sparsity * alpha, no. of additive and subtractive iter.
    """
    # if sdr is already at target sparsity.
    if (sum(SDRT) / len(SDRT)) <= target_sparsity:
        return blist2list(SDRT), 0, 0

    # assume SDR is binary list
    type = 0
    if type == 1:
        rng = random.Random()
        rng.seed(100)
    else:
        idx_perm = 0

    #    SDRT = list2blist(bSDR)
    N = len(SDRT)
    SDR_FINAL = [False] * N
    PKZ = list(SDRT)

    NK0 = 0
    NK1 = 0

    # additive phase: add bits until target sparsity is reached
    while (sum(SDR_FINAL) / N < target_sparsity):
        if type == 1:
            rng.shuffle(PKZ)
        else:
            PKZ[:] = [PKZ[i] for i in permutation[idx_perm]]
            idx_perm = 1 if idx_perm == 0 else 0

        SDR_FINAL = or_blist(SDR_FINAL, and_blist(SDRT, PKZ))
        NK1 = NK1 + 1

    #print(f"sparsity end of additive {sum(SDR_FINAL) / N}")

    while (sum(SDR_FINAL) / N > target_sparsity * alpha):
        if type == 1:
            rng.shuffle(PKZ)
        else:
            PKZ[:] = [PKZ[i] for i in permutation[idx_perm]]
            idx_perm = 1 if idx_perm == 0 else 0

        SDR_FINAL = and_blist(SDR_FINAL, not_blist(PKZ))
        NK0 = NK0 + 1

    #print(f"sparsity end of substructive {sum(SDR_FINAL) / N}")

    return blist2list(SDR_FINAL), NK0, NK1


def encode_sequence(SDR_SEQ, permutation):
    """
    step 2 of TSSE.
    """
    # assume SDR_SEQ is binary list
    #    rng = random.Random()
    #    rng.seed(seed_val)
    N = len(SDR_SEQ[0])
    #    permutation = rng.shuffle(list(range(N)))

    SDR_FINAL = [False] * N

    # for 3 sdrs the final sdr is sdr[0] + sdr[1]*p + sdr[2]*p*p
    for idx, sdr in enumerate(SDR_SEQ):
        for i in range(idx + 1):
            sdr[:] = [sdr[j] for j in permutation]

        SDR_FINAL = or_blist(SDR_FINAL, sdr)

    return SDR_FINAL


def pad_binary_list(binary_list, N):
    """
    Pads a binary list with False values until it reaches length N.
    Note: len(binary_list) <= N
    """      
    return binary_list + [False] * (N - len(binary_list))


#### Fast TSSE  - improves performance by 4x (empirically) ####
import numpy as np

def encode_sequence_fast(SDR_SEQ, permutation):
    """
    step 2 of TSSE using numpy for faster operations
    """
    N = len(SDR_SEQ[0])
    # Convert to numpy array for faster operations
    np_permutation = np.array(permutation)
    SDR_FINAL = np.zeros(N, dtype=bool)
    
    for idx, sdr in enumerate(SDR_SEQ):
        np_sdr = np.array(sdr, dtype=bool)
        # Apply permutations
        for i in range(idx + 1):
            np_sdr = np_sdr[np_permutation]
        # OR operation with numpy
        SDR_FINAL = np.logical_or(SDR_FINAL, np_sdr)
    
    return SDR_FINAL.tolist()

def stable_cdt_fast(SDRT, target_sparsity, permutation, alpha=1.1):
    """
    step 3 of TSSE using numpy for faster operations
    """
    N = len(SDRT)
    np_SDRT = np.array(SDRT, dtype=bool)
    current_sparsity = np.mean(np_SDRT)
    
    if current_sparsity <= target_sparsity:
        return SDRT, 0, 0
    
    np_permutation = [np.array(p) for p in permutation]
    SDR_FINAL = np.zeros(N, dtype=bool)
    PKZ = np_SDRT.copy()
    NK0 = NK1 = 0
    idx_perm = 0
    
    # Additive phase
    while np.mean(SDR_FINAL) < target_sparsity:
        PKZ = PKZ[np_permutation[idx_perm]]
        idx_perm = 1 if idx_perm == 0 else 0
        SDR_FINAL = np.logical_or(SDR_FINAL, np.logical_and(np_SDRT, PKZ))
        NK1 += 1
    
    # Subtractive phase
    while np.mean(SDR_FINAL) > target_sparsity * alpha:
        PKZ = PKZ[np_permutation[idx_perm]]
        idx_perm = 1 if idx_perm == 0 else 0
        SDR_FINAL = np.logical_and(SDR_FINAL, ~PKZ)
        NK0 += 1
    
    return SDR_FINAL.tolist(), NK0, NK1


def test_cdt():
    # SDR list:
    sdr_val = list()  # SDR.sparse
    sdr_bin_list = list()  # SDR.dense
    rng = random.Random()
    N = 2048  # encoder size
    bits = 41  # active bits
    rng.seed(10)
    sparsity = 0.02  # active bits = 41
    permutation_cdt = list()  # needs to have 2 elements
    permutation_enc = list(range(N))
    rng.shuffle(permutation_enc)  # without step 2
    permutation_cdt.append(list(range(N)))
    permutation_cdt.append(list(range(N)))
    rng.shuffle(permutation_cdt[0])
    rng.shuffle(permutation_cdt[1])

    for i in range(16):
        # SDR(N): creates an SDR with N bits.
        # use: SDR.sparse = [list of bits : idxs], SDR.dense = [binary rep.]
        sdr_val.append(SDR(N))
        sdr_val[i].sparse = list(range(20 + i, 61 + i))

    print(f'sdr_val len: {len(sdr_val)}')

    # step 1 of TSSE
    for i in range(16):
        sdr_bin_list.append(SDR2blist(sdr_val[i]))  # boolean list of each SDR

    print(f'sdr_bin_list len: {len(sdr_bin_list)}')

    sdr_encoded_bin = encode_sequence_fast(sdr_bin_list, permutation_enc)  # step 2 of TSSE
    # we are left with a single SDR, that is the OR of all the SDRs encoding + permutation

    print(f'sparsity: {sum(sdr_encoded_bin) / len(sdr_encoded_bin)}')  # 0.275, way higher than target sparsity

    sdr_encoded = blist2SDR(sdr_encoded_bin)
    sdr_cdt_bin, N0, N1 = stable_cdt_fast(sdr_encoded_bin, sparsity, permutation_cdt)
    print(f'sparsity: {sum(sdr_cdt_bin) / len(sdr_cdt_bin)}')  # 0.0167, 

    sdr_cdt = blist2SDR(sdr_cdt_bin)

    print(f"size: {sdr_cdt.size}, N0 {N0}, {N1}")


def test_multi_channel_cdt():
    sdr_val = list()
    sdr_bin_list = list()  # TSSE buffer
    rng = random.Random()
    rng.seed(10)
    N_channel1 = 8
    N_channel2 = 16
    N = max(N_channel1, N_channel2)
    sparsity = (2/16)
    permutation_cdt = list()  # needs to have 2 elements
    permutation_enc = list(range(N))
    rng.shuffle(permutation_enc)  # for step 2
    permutation_cdt.append(list(range(N)))
    permutation_cdt.append(list(range(N)))
    rng.shuffle(permutation_cdt[0])
    rng.shuffle(permutation_cdt[1])

    sdr_val.append(SDR(N_channel1))
    sdr_val.append(SDR(N_channel2))
    sdr_val[0].sparse = list(range(2,4))  # encoding
    sdr_val[1].sparse = list(range(5,8))  # encoding

    sdr_bin_list.append(SDR2blist(sdr_val[0]))
    sdr_bin_list.append(SDR2blist(sdr_val[1]))
    print(f'sdr_bin_list: {sdr_bin_list}')

    for i in range(2):  # padding
        sdr_bin_list[i] = pad_binary_list(sdr_bin_list[i], N)
    print(f'sdr_bin_list after padding:\n{sdr_bin_list}')

    sdr_encoded_bin = encode_sequence_fast(sdr_bin_list, permutation_enc)
    print(f'\nsdr_encoded_bin: {sdr_encoded_bin}')
    print(f'before cdt:sparsity: {sum(sdr_encoded_bin) / len(sdr_encoded_bin)}') 

    sdr_encoded = blist2SDR(sdr_encoded_bin)
    sdr_cdt_bin, N0, N1 = stable_cdt_fast(sdr_encoded_bin, sparsity, permutation_cdt)
    print(f'after cdt: sparsity: {sum(sdr_cdt_bin) / len(sdr_cdt_bin)}')

    sdr_cdt = blist2SDR(sdr_cdt_bin)

    print(f"{sdr_cdt_bin}, N0 {N0}, {N1}")


def main():
    test_cdt()
    print('test_cdt done\n\n')
    test_multi_channel_cdt()
    print('test_multi_channel_cdt done')
    # print("running..")


if __name__ == "__main__":
    #test_stage_id_to_global_id()

    # test_get_state_sdr()
    # test_calc_anomaly_stats()
    #test_count_continuous_ones()
    main()

