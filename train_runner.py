import os, sys, json, itertools, time
from multiprocessing import Process


def get_free_gpu_id():
    """
    Get the first free GPU
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [
        int(x.split()[2]) for x in open('tmp', 'r').readlines()
    ]
    return memory_available.index(max(memory_available))


def get_conf_enumeration():
    '''
    all_models = ['hp', 'hpi', 'hpic']
    all_vote_thresholds = [
        0, 0.25, 0.5, 0.75, 1, 1.25,1.5, 1.75, 2
    ]'''
    all_models = ['c']
    all_vote_thresholds = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    conf_v = list(itertools.product(all_models, all_vote_thresholds))
    return conf_v


pool = []
for conf_v in get_conf_enumeration():
    conf = {
        'models': conf_v[0],
        'vote_threshold': conf_v[1],
        'device': 'cuda:' + str(get_free_gpu_id())
    }
    print(conf)
    args = '\'' + json.dumps(conf) + '\''
    p = Process(target=os.system, args=('python train.py ' + args, ))
    p.start()
    time.sleep(30)
for p in pool:
    p.join()
