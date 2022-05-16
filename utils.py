import logging, json, time, os, random
import torch, am_decoder
import numpy as np
import zipfile, json, re, time, amd
from io import BytesIO
from axmlparser import AXML

from torch.utils.data import DataLoader

amdr = amd.AndroidXMLDecompress()


def extract_apk(path):
    try:
        apk = zipfile.ZipFile(path, 'r')
    except zipfile.BadZipFile:
        raise Exception("{} is not a valid APK file".format(path))
    try:
        #am_file = apk.read('AndroidManifest.xml')
        am_file = BytesIO(apk.read('AndroidManifest.xml'))
        #print(apk.read('AndroidManifest.xml'))
    except:
        am_file = None
    try:
        classes_file = apk.read('classes.dex')
        #classes_file = BytesIO(apk.read('classes.dex'))
    except:
        classes_file = None
    return am_file, classes_file, os.path.getsize(path)


def get_entropy(x, block_size=256):
    left = len(x) % block_size
    if left == 0:
        pass
    elif left < block_size / 2:
        x = x[:-left]
    else:
        x = np.pad(x, (0, block_size - left), 'constant')
    x = x.reshape([-1, block_size])

    nrows, ncols = x.shape
    Hlist = []
    for row in x:
        counts = np.bincount(row)
        p = counts / float(ncols)
        p = p[p != 0]
        Hlist.append(-np.sum(p * np.log2(p), axis=0))
    return np.array(Hlist)


def get_ip_feature(am_file, dictionary):
    words = np.zeros(len(dictionary))
    try:
        data = am_decoder.decode(am_file)
    except:
        return words
    #data = amdr.decompressXML(am_file.read())
    for key in re.findall(r'android:name=\"(.*)\"', data):
        if key in dictionary:
            words[dictionary[key]] += 1
    return words


class AverageMeter(object):
    """Computes and stores the average and current value.

    >>> acc = AverageMeter()
    >>> acc.update(0.6)
    >>> acc.update(0.8)
    >>> print(acc.avg)
    0.7
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BestSaver(object):
    """Save pytorch model with best performance. Template for save_path is:

        model_{save_path}_{comment}.pth

    >>> comment = "v1"
    >>> saver = BestSaver(comment)
    >>> auc = 0.6
    >>> saver.save(0.6, model.state_dict())
    """
    def __init__(self, comment=None):
        # Get current executing script name
        import __main__, os
        exe_fname = os.path.basename(__main__.__file__)
        base_path = os.path.split(__main__.__file__)[0]

        save_path = os.path.join(
            base_path, "models/model_{}".format(exe_fname.split(".")[0]))

        if comment is not None and str(comment):
            save_path = save_path + "_" + str(comment)

        #save_path = save_path + ".pth"

        self.save_path = save_path
        self.best = float('-inf')

    def save(self, metric, data, epoch):
        if metric > self.best:
            self.best = metric
            torch.save(data, self.save_path + ".pth")
            logging.info("Saved best model to {}".format(self.save_path))


def config_logging(comment=None):
    """Configure logging for training log. The format is 

        `log_{log_fname}_{comment}.log`

    .g. for `train.py`, the log_fname is `log_train.log`.
    Use `logging.info(...)` to record running log.

    Args:
        comment (any): Append comment for log_fname
    """

    # Get current executing script name
    import __main__, os
    exe_fname = os.path.basename(__main__.__file__)
    log_fname = "log_{}".format(exe_fname.split(".")[0])

    if comment is not None and str(comment):
        log_fname = log_fname + "_" + str(comment)

    log_fname = log_fname + time.strftime(" %y-%m-%d %H:%M:%S",
                                          time.localtime()) + ".log"
    log_fname = os.path.dirname(os.path.abspath(
        __main__.__file__)) + '/logs/' + log_fname
    log_format = "%(asctime)s [%(levelname)-5.5s] %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.FileHandler(log_fname),
                  logging.StreamHandler()])


class JsonLogger():
    def __init__(self, models, vote_threshold):
        import __main__, os
        log_fname = "log_{}".format(models)
        log_fname = log_fname + f"_adav{vote_threshold}"

        log_fname = log_fname + time.strftime(" %y-%m-%d %H:%M:%S",
                                              time.localtime()) + ".json"
        self.log_fname = os.path.dirname(os.path.abspath(
            __main__.__file__)) + '/logs_json/' + log_fname
        self.log_dict = {}

    def log(self, key, value):
        self.log_dict[key] = value

    def get(self, key):
        if key not in self.log_dict:
            return None
        return self.log_dict[key]

    def append(self, key, value):
        if key not in self.log_dict:
            self.log_dict[key] = []
        self.log_dict[key].append(value)

    def save(self):
        with open(self.log_fname, 'w') as f:
            json.dump(self.log_dict, f)


def remove_invalid_apks(path):
    for root, _, files in os.walk(path):
        for file in files:
            print(file)
            file_path = os.path.join(root, file)
            try:
                apk = zipfile.ZipFile(file_path, 'r')
                if apk.read('classes.dex') is None or apk.read(
                        'AndroidManifest.xml') is None:
                    1 / 0
            except:
                print("{} is not a valid APK file".format(file_path))
                os.remove(file_path)


def get_dataloaders(dataset=None,
                    collate_fn=None,
                    ben_path='',
                    mal_path='',
                    dict_path='',
                    train_val_test=[0.8, 0.1, 0.1],
                    batch_size=32,
                    num_workers=4,
                    fs=1.0):
    dictionary = json.load(open(dict_path, 'r'))

    data = []
    for root, _, files in os.walk(ben_path):
        for file in files:
            data.append((os.path.join(root, file), False))
    ben_size = len(data)
    print('Load Benign Samples:', ben_size)
    for root, _, files in os.walk(mal_path):
        for file in files:
            data.append((os.path.join(root, file), True))
    mal_size = len(data) - ben_size

    #data = data[:-(mal_size - ben_size + 1000)]
    print('Load Malicious Samples:', len(data) - ben_size)

    random.shuffle(data)
    split_1 = int(len(data) * train_val_test[0])
    split_2 = int(len(data) * (train_val_test[0] + train_val_test[1]))
    train_data = data[:split_1]
    val_data = data[split_1:split_2]
    test_data = data[split_2:]

    train_dataset = dataset(train_data, dictionary=dictionary, fs=fs)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
    val_dataset = dataset(val_data, dictionary=dictionary, fs=fs)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    test_dataset = dataset(test_data, dictionary=dictionary, fs=fs)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader


def prepare_dataloaders(
        data_path='/home/user1/projects/hubochao/datasets/my/dataset_all.json',
        dataset=None,
        train_size=0.8,
        val_size=0.1,
        batch_size=32,
        num_workers=1,
        fs=1.0,
        collate_fn=None):
    dict_len = len(
        json.load(open('/home/user1/projects/hubochao/datasets/my/dict.json')))
    meta_data = json.load(open(data_path, 'r'))
    split_1 = int(train_size * len(meta_data))
    split_2 = int(split_1 + val_size * len(meta_data))

    train_list = meta_data[:split_1]
    val_list = meta_data[split_1:split_2]
    test_list = meta_data[split_2:]

    train_list += val_list[:int(len(val_list) * 0.2)]
    train_list += test_list[:int(len(test_list) * 0.2)]

    train_dataset = dataset(train_list, fs=fs, type='train', dict_len=dict_len)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
    val_dataset = dataset(val_list, fs=fs, type='valid', dict_len=dict_len)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    test_dataset = dataset(test_list, fs=fs, type='test', dict_len=dict_len)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             collate_fn=collate_fn)
    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, (
        split_1, split_2 - split_1, len(meta_data) - split_2)


def test():
    count = 0
    failed1 = 0
    failed2 = 0
    for root, _, files in os.walk('/home/user1/projects/hubochao/datasets/'):
        for file in files:
            apk = zipfile.ZipFile(os.path.join(root, file), 'r')
            am = apk.read('AndroidManifest.xml')
            try:
                data = am_decoder.decode(BytesIO(am))
            except:
                failed1 += 1
                print(file)
            try:
                axml = AXML(am)
                axml.getPackageName()
                #axml.printAll()
            except:
                failed2 += 1
                print(file)
            count += 1
            print(count, failed1, failed2)
