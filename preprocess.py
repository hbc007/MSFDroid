import os, am_decoder, shutil, math, time, json, random, multiprocessing
import PIL.Image as Image, os, sys, array, math, re, collections
import numpy as np
from collections import Counter
import math, scipy.signal
import matplotlib.pyplot as plt

#source_dir = '/home/user1/projects/hubochao/datasets/MalDroid-2020/'
#source_dir = '/home/user1/projects/hubochao/datasets/MalDroid-2020-Benign'
#source_dir = '/home/user1/projects/hubochao/datasets/CICInvesAndMal2019'
source_dir = '/home/user1/projects/hubochao/datasets/CICInvesAndMal2019-Benign'
target_dir = '/home/user1/projects/hubochao/datasets/my'
thread_num = 60


#定义dex_magic的类。用于表示dex魔数
class struct_dex_magic:
    dex = [3]  #dex标志位
    newline = [1]
    ver = [3]  #dex版本信息
    zero = [1]

    def get(self):
        return str(self.dex + self.newline + self.ver + self.zero)


#定义dex_hander的类。用于表示dex头部信息，
class struct_dex_header:
    magic = struct_dex_magic()  #dex魔数，起始：0x00，长度 8
    checksum = [4]  #校验值，起始：0x8,长度 4。采用alder-32算法，将0xc到文件结尾所有的byte数据计算
    signature = [20]  #签名信息，起始：0xc,长度 0x14。采用sha1算计，将0x20到文件结尾所有的byte数据计算
    file_size = [4]  #dex文件大小，起始:0x20,长度 4
    header_size = [4]  #文件头大小，起始：0x24，长度 4
    endian_tag = [4]  #文件字节序，起始：0x28，长度 4。默认小尾字节序，默认数据是: 0x78 0x56 0x34 0x12
    link_size = [4]  #文件链接段大小，起始：0x2c,长度 4，如果数值为0表示静态链接
    link_off = [4]  #文件链接段偏移，起始：0x30，长度 4，
    map_off = [4]  #map数据偏移，起始：0x34，长度4
    string_ids_size = [4]  #字符串的数量，起始：0x38，长度4，即string_id_item的数量
    string_ids_off = [4]  #字符串偏移，起始：0x3c,长度4，即string_id_item的起始位置
    type_ids_size = [4]  #类的数量，起始：0x40，长度4，即type_id_item的数量
    type_ids_off = [4]  #类的偏移，起始：0x44，长度4，即typy_id_item的起始位置
    proto_ids_size = [4]  #方法原型的数量，起始：0x48，长度4，即proto_id_item的数量
    proto_ids_off = [4]  #方法原型的偏移，起始：0x4c,长度4，即proto_id_item的起始位置
    field_ids_size = [4]  #字段的数量，起始：0x50，长度4，即field_id_item的数量
    field_ids_off = [4]  #字段的偏移，起始：0x54，长度4，即field_id_item的起始位置
    method_ids_size = [4]  #方法的数量，起始：0x58,长度4，即method_id_item的数量
    method_ids_off = [4]  #方法的偏移，起始：0x5c,长度4，即method_id_item的起始位置
    class_defs_size = [4]  #类定义的数量，起始：0x60,长度4，即class_def_item的数量
    class_defs_off = [4]  #类定义的偏移，起始：0x64,长度4，即class_def_item的起始位置
    data_size = [4]  #数据段 的大小，起始：0x68,长度4
    data_off = [4]  #数据段 的偏移，起始：0x6c,长度4

    def get_dict(self):
        return {
            'magic': self.magic.get(),
            'checksum': self.checksum,
            'signature': self.signature,
            'file_size': self.file_size,
            'header_size': self.header_size,
            'endian_tag': self.endian_tag,
            'link_size': self.link_size,
            'link_off': self.link_off,
            'map_off': self.map_off,
            'string_ids_size': self.string_ids_size,
            'string_ids_off': self.string_ids_off,
            'type_ids_size': self.type_ids_size,
            'type_ids_off': self.type_ids_off,
            'proto_ids_size': self.proto_ids_size,
            'proto_ids_off': self.proto_ids_off,
            'field_ids_size': self.field_ids_size,
            'field_ids_off': self.field_ids_off,
            'method_ids_size': self.method_ids_size,
            'method_ids_off': self.method_ids_off,
            'class_defs_size': self.class_defs_size,
            'class_defs_off': self.class_defs_off,
            'data_size': self.data_size,
            'data_off': self.data_off,
        }


def isDex(dexFileMmap):
    dexFileMmap.seek(0)
    dexMagic = dexFileMmap.read(8)
    if (dexMagic.hex() !=
            "6465780a30333500"):  #判断文件格式是否为dex,"64 65 78 0A 30 33 35 00"
        return False
    else:
        return True


#读取dexheader数据
def parseDexHeader(dexFileMmap):
    #将hex数据移位拼接，例如 0x11 +0x22 +0x33 +0x44 = 0x11223344
    def append_hex(arg0, arg1, arg2, arg3):
        arg0 = arg0 << 24
        arg1 = arg1 << 16
        arg2 = arg2 << 8
        result = arg0 + arg1 + arg2 + arg3
        #return hex(result)
        return result

    dex_header = struct_dex_header()  #dex结构体
    dexFileMmap.seek(0)
    dexheader = dexFileMmap.read(0x70)  #dexheader长度固定，0x70,112个字节

    dex_magic = struct_dex_magic()

    #dex_header数据获取，同时将数据整型 大小端转换 转为可用格式
    dex_magic.dex = dexheader[0:3]
    dex_magic.newline = dexheader[3:4]
    dex_magic.ver = dexheader[4:7]
    dex_magic.zero = dexheader[7:8]

    dex_header.magic = dex_magic

    dex_header.checksum = dexheader[8:0xc]
    tmpValue = list(reversed(dex_header.checksum))
    dex_header.checksum = append_hex(tmpValue[0], tmpValue[1], tmpValue[2],
                                     tmpValue[3])

    dex_header.signature = dexheader[0xc:0x20]
    dex_header.signature = dex_header.signature.hex()

    dex_header.file_size = dexheader[0x20:0x24]
    tmpValue = list(reversed(dex_header.file_size))
    dex_header.file_size = append_hex(tmpValue[0], tmpValue[1], tmpValue[2],
                                      tmpValue[3])

    dex_header.header_size = dexheader[0x24:0x28]
    tmpValue = list(reversed(dex_header.header_size))
    dex_header.header_size = append_hex(tmpValue[0], tmpValue[1], tmpValue[2],
                                        tmpValue[3])

    dex_header.endian_tag = dexheader[0x28:0x2c]
    tmpValue = list(reversed(dex_header.endian_tag))
    dex_header.endian_tag = append_hex(tmpValue[0], tmpValue[1], tmpValue[2],
                                       tmpValue[3])

    dex_header.link_size = dexheader[0x2c:0x30]
    tmpValue = list(reversed(dex_header.link_size))
    dex_header.link_size = append_hex(tmpValue[0], tmpValue[1], tmpValue[2],
                                      tmpValue[3])

    dex_header.link_off = dexheader[0x30:0x34]
    tmpValue = list(reversed(dex_header.link_off))
    dex_header.link_off = append_hex(tmpValue[0], tmpValue[1], tmpValue[2],
                                     tmpValue[3])

    dex_header.map_off = dexheader[0x34:0x38]
    tmpValue = list(reversed(dex_header.map_off))
    dex_header.map_off = append_hex(tmpValue[0], tmpValue[1], tmpValue[2],
                                    tmpValue[3])

    dex_header.string_ids_size = dexheader[0x38:0x3c]
    tmpValue = list(reversed(dex_header.string_ids_size))
    dex_header.string_ids_size = append_hex(tmpValue[0], tmpValue[1],
                                            tmpValue[2], tmpValue[3])

    dex_header.string_ids_off = dexheader[0x3c:0x40]
    tmpValue = list(reversed(dex_header.string_ids_off))
    dex_header.string_ids_off = append_hex(tmpValue[0], tmpValue[1],
                                           tmpValue[2], tmpValue[3])

    dex_header.type_ids_size = dexheader[0x40:0x44]
    tmpValue = list(reversed(dex_header.type_ids_size))
    dex_header.type_ids_size = append_hex(tmpValue[0], tmpValue[1],
                                          tmpValue[2], tmpValue[3])

    dex_header.type_ids_off = dexheader[0x44:0x48]
    tmpValue = list(reversed(dex_header.type_ids_off))
    dex_header.type_ids_off = append_hex(tmpValue[0], tmpValue[1], tmpValue[2],
                                         tmpValue[3])

    dex_header.proto_ids_size = dexheader[0x48:0x4c]
    tmpValue = list(reversed(dex_header.proto_ids_size))
    dex_header.proto_ids_size = append_hex(tmpValue[0], tmpValue[1],
                                           tmpValue[2], tmpValue[3])

    dex_header.proto_ids_off = dexheader[0x4c:0x50]
    tmpValue = list(reversed(dex_header.proto_ids_off))
    dex_header.proto_ids_off = append_hex(tmpValue[0], tmpValue[1],
                                          tmpValue[2], tmpValue[3])

    dex_header.field_ids_size = dexheader[0x50:0x54]
    tmpValue = list(reversed(dex_header.field_ids_size))
    dex_header.field_ids_size = append_hex(tmpValue[0], tmpValue[1],
                                           tmpValue[2], tmpValue[3])

    dex_header.field_ids_off = dexheader[0x54:0x58]
    tmpValue = list(reversed(dex_header.field_ids_off))
    dex_header.field_ids_off = append_hex(tmpValue[0], tmpValue[1],
                                          tmpValue[2], tmpValue[3])

    dex_header.method_ids_size = dexheader[0x58:0x5c]
    tmpValue = list(reversed(dex_header.method_ids_size))
    dex_header.method_ids_size = append_hex(tmpValue[0], tmpValue[1],
                                            tmpValue[2], tmpValue[3])

    dex_header.method_ids_off = dexheader[0x5c:0x60]
    tmpValue = list(reversed(dex_header.method_ids_off))
    dex_header.method_ids_off = append_hex(tmpValue[0], tmpValue[1],
                                           tmpValue[2], tmpValue[3])

    dex_header.class_defs_size = dexheader[0x60:0x64]
    tmpValue = list(reversed(dex_header.class_defs_size))
    dex_header.class_defs_size = append_hex(tmpValue[0], tmpValue[1],
                                            tmpValue[2], tmpValue[3])

    dex_header.class_defs_off = dexheader[0x64:0x68]
    tmpValue = list(reversed(dex_header.class_defs_off))
    dex_header.class_defs_off = append_hex(tmpValue[0], tmpValue[1],
                                           tmpValue[2], tmpValue[3])

    dex_header.data_size = dexheader[0x68:0x6c]
    tmpValue = list(reversed(dex_header.data_size))
    dex_header.data_size = append_hex(tmpValue[0], tmpValue[1], tmpValue[2],
                                      tmpValue[3])

    dex_header.data_off = dexheader[0x6c:0x70]
    tmpValue = list(reversed(dex_header.data_off))
    dex_header.data_off = append_hex(tmpValue[0], tmpValue[1], tmpValue[2],
                                     tmpValue[3])

    #调用打印函数，打印dex文件头数据
    #dex_header.printInfo()
    return dex_header.get_dict()


def writeParseHeader(target_dir):
    files = getAllFile(target_dir, ext='.dex')
    threading_pool = []
    header_list = []
    for file in files:
        with open(file, 'rb') as f:
            if not isDex(f):
                continue
            header = parseDexHeader(f)
            header_list.append(header)
    with open(target_dir.rstrip('/') + '_header.json', 'w') as f:
        json.dump(header_list, f)


def getBytes(file_path):
    fileobj = open(file_path, mode='rb')
    buffer = array.array('B', fileobj.read())  #二进制(8个一)占大小为一个字节，换成十进制变成255
    size = len(buffer)
    fileobj.close()
    return np.array(buffer), size


def _calImgWidth(size):
    size /= 1024
    width = 512
    if size < 10:
        width = 16
    elif size < 30:
        width = 32
    elif size < 30:
        width = 32
    elif size < 60:
        width = 64
    elif size < 100:
        width = 128
    elif size < 200:
        width = 192
    elif size < 500:
        width = 256
    elif size < 1000:
        width = 384
    return width


def getBytesMatrix(buffer, buffer_size):
    #width = _calImgWidth(buffer_size)
    width = math.ceil(math.sqrt(math.ceil(buffer_size / 3)))
    height = width
    img = np.pad(buffer, (0, width * height * 3 - buffer_size),
                 'constant').reshape((height, width, 3))
    return Image.fromarray(img.astype('uint8')).convert('RGB')


def getLocMatrix(buffer, buffer_size):
    b = np.pad(buffer, (0, 0 if buffer_size % 5 == 0 else 5 - buffer_size % 5),
               'constant')
    img = np.zeros((256, 256, 3), dtype=np.int)
    for i in range(0, len(b) - 1, 5):
        img[b[i], b[i + 1], 0] = (img[b[i], b[i + 1], 0] + b[i + 2]) % 255
        img[b[i], b[i + 1], 1] = (img[b[i], b[i + 1], 1] + b[i + 3]) % 255
        img[b[i], b[i + 1], 2] = (img[b[i], b[i + 1], 2] + b[i + 4]) % 255
    return Image.fromarray(img.astype('uint8')).convert('RGB')


def getMarkovMatrix(buffer, buffer_size):
    img = np.zeros((256, 256), dtype=np.int)
    for i in range(buffer_size - 2):
        img[buffer[i], buffer[i + 1]] += 1
    return Image.fromarray(img.astype('uint8')).convert('RGB')


def getEntropyMatrix(buffer, buffer_size, block_size=256):
    length = buffer_size / block_size
    left = buffer_size % block_size
    buffer = buffer[:-left] if left < block_size / 2 else np.pad(
        buffer, (0, block_size - left), 'constant')
    buffer = buffer.reshape([-1, block_size])
    Hlist = []
    for block in buffer:
        H = 0
        for x, count in Counter(block).items():
            px = count / block_size
            hx = px * math.log2(px)
            H += hx
        H = -H
        Hlist.append(H)
    Hlist = np.array(Hlist)
    return Hlist


def getDexHeaderMatrix(buffer, buffer_size):
    headerSize = 0x70
    header = buffer[:headerSize]
    header = header / 256
    return header


def getDexSplit(buffer, buffer_size, params):
    stype = params.get('type')
    path = params.get('path')
    f = open(path, 'rb')
    dtypes = parseDexHeader(f)
    ssize = 0
    soff = 0
    try:
        ssize = dtypes[stype + '_size']
        soff = dtypes[stype + '_off']
        b = buffer[soff:soff + ssize]
        img = getBytesMatrix(b, len(b))
        f.close()
        return img
    except:
        print(
            f'getDexSplit Error: [{soff}:{soff + ssize}] len:{ssize}/{buffer_size}',
            path)
        f.close()
        return None


def buffer2Matrix(buffer, buffer_size):  #将文件转换为概率矩阵的方法
    bm = getBytesMatrix(buffer, buffer_size)
    lm = getLocMatrix(buffer, buffer_size)
    mm = getMarkovMatrix(buffer, buffer_size)
    return bm, lm, mm


def getAllFile(file_path, ext=''):
    files = []
    for root, _, file_names in os.walk(file_path):
        for file_name in file_names:
            if ext == None:
                files.append(os.path.join(root, file_name))
            elif os.path.splitext(file_name)[1] == ext:
                files.append(os.path.join(root, file_name))
    return files


def extract(apk_file):
    target_path = os.path.join(target_dir,
                               os.path.basename(os.path.splitext(apk_file)[0]))
    os.system(
        f'unzip -o -jd {target_path} {apk_file} "*.dex" "AndroidManifest.xml"'
        #f'unzip -o -jd {target_path} {apk_file} "*.dex" "*.so" "*.jar" "AndroidManifest.xml"'
    )
    try:
        androidManifestDecode(os.path.join(target_path, "AndroidManifest.xml"))
        saveDexImages(target_path)
    except:
        pass


def saveDexImages(file_path):
    for file_name in os.listdir(file_path):
        file_name, ext = os.path.splitext(file_name)
        if ext == '.dex':
            buffer, size = getBytes(os.path.join(file_path,
                                                 file_name + '.dex'))
            #print(f'{file_name}.dex: {"{:.2f}".format(size/1024)}KB')
            bm, lm, mm = buffer2Matrix(buffer, size)
            bm.save(os.path.join(file_path, f'{file_name}_byte.png'))
            lm.save(os.path.join(file_path, f'{file_name}_loc.png'))
            mm.save(os.path.join(file_path, f'{file_name}_markov.png'))
            #print('image saved:', file_path)


def androidManifestDecode(manifest_file, remove_err=True):
    file = open(manifest_file, 'rb')
    try:
        data = am_decoder.decode(file)
        file.close()
        file = open(manifest_file, 'w')
        file.write(data)
        file.close()
        #print('manifest decoded:', manifest_file)
    except:
        file.close()
        print('Error:', manifest_file)
        if remove_err:
            shutil.rmtree(os.path.dirname(manifest_file))


def generateImage(name,
                  target_dir,
                  bufferProcessFunc,
                  thread_num=16,
                  gen=True,
                  params=None):
    files = getAllFile(target_dir, ext='.dex')
    threading_pool = []

    class Process(multiprocessing.Process):
        def __init__(self, files, threadId):
            multiprocessing.Process.__init__(self)
            self.files = files
            self.threadId = threadId
            self.daemon = True

        def run(self):
            i = 0
            for file in self.files:
                buffer, size = getBytes(file)
                if params is not None:
                    params['path'] = file
                    img = bufferProcessFunc(buffer, size, params)
                else:
                    img = bufferProcessFunc(buffer, size)
                target_filename = os.path.splitext(
                    file)[0] + '_' + name + '.png'
                #print('Thread', self.threadId, 'saved:', target_filename)
                if gen and img:
                    img.save(target_filename)
                if True and i % 5 == 0:
                    print('Thread', self.threadId, 'saved:', target_filename)
                    print(
                        f'\033[1;32mThread {self.threadId}: {"{:.3f}".format((i+1)/len(self.files)*100)}% completed \033[0m'
                    )
                i += 1

    for i in range(thread_num):
        threading_pool.append(
            Process(
                files[math.floor(i / thread_num *
                                 len(files)):math.floor((i + 1) / thread_num *
                                                        len(files))], i))
        threading_pool[-1].start()
        time.sleep(0.05)
    for thread in threading_pool:
        thread.join()


def generateNumpy(name,
                  target_dir,
                  bufferProcessFunc,
                  thread_num=16,
                  gen=True,
                  params=None):
    files = getAllFile(target_dir, ext='.dex')
    threading_pool = []

    class Process(multiprocessing.Process):
        def __init__(self, files, threadId):
            multiprocessing.Process.__init__(self)
            self.files = files
            self.threadId = threadId
            self.daemon = True

        def run(self):
            i = 0
            for file in self.files:
                buffer, size = getBytes(file)
                if params is not None:
                    params['path'] = file
                    data = bufferProcessFunc(buffer, size, params)
                else:
                    data = bufferProcessFunc(buffer, size)
                target_filename = os.path.splitext(
                    file)[0] + '_' + name + '.npy'
                #print('Thread', self.threadId, 'saved:', target_filename)
                if gen and data:
                    np.save(target_filename, data)
                if True and i % 5 == 0:
                    print('Thread', self.threadId, 'saved:', target_filename)
                    print(
                        f'\033[1;32mThread {self.threadId}: {"{:.3f}".format((i+1)/len(self.files)*100)}% completed \033[0m'
                    )
                i += 1

    for i in range(thread_num):
        threading_pool.append(
            Process(
                files[math.floor(i / thread_num *
                                 len(files)):math.floor((i + 1) / thread_num *
                                                        len(files))], i))
        threading_pool[-1].start()
        time.sleep(0.05)
    for thread in threading_pool:
        thread.join()


def generateDict(
        path,
        dictpath='/home/user1/projects/hubochao/datasets/my/dict_src.json'):
    dictionary = collections.Counter(dict())
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            if not os.path.splitext(file_name)[1] == '.xml':
                continue
            with open(os.path.join(root, file_name), 'r') as f:
                xml = ''.join(f.readlines())
                key = re.findall(r'android:name=\"(.*)\"', xml)
                val = [1] * len(key)
                dictionary = dictionary + collections.Counter(
                    dict(zip(key, val)))
    dictionary = dict(
        sorted(dictionary.items(), key=lambda kv: (kv[1], kv[0]),
               reverse=True))
    with open(dictpath, 'w') as f:
        json.dump(dictionary, f)


def generateBOW(path, dict_path):
    dictionary = json.load(open(dict_path, 'r'))
    print(len(dictionary))
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            if not os.path.splitext(file_name)[1] == '.xml':
                continue
            with open(os.path.join(root, file_name), 'r') as f:
                xml = ''.join(f.readlines())
                words = np.zeros(len(dictionary))
                for key in re.findall(r'android:name=\"(.*)\"', xml):
                    if key in dictionary:
                        words[dictionary[key]] += 1
                np.save(os.path.join(root, 'intent_permission.npy'), words)


def updateDictionary(src_path, dst_path, abort=3):
    dic = {}
    with open(src_path, 'r') as f:
        dic = json.load(f)
        dic = ['UNK'] + [
            x for x in dic if dic[x] > abort or x.find('android.intent') == 0
            or x.find('android.permission') == 0
        ]
        dic = dict(zip(dic, list(range(0, len(dic)))))
    with open(dst_path, 'w') as f:
        json.dump(dic, f)


def updatePath():
    dataset = json.load(
        open('/home/user1/projects/hubochao/datasets/my/dataset_all.json',
             'r'))

    is_mal_list = {}
    for apk_info in dataset:
        is_mal_list[apk_info['name']] = apk_info['is_mal']
    print(len(is_mal_list))
    path_list = {}
    test_dataset = {}
    for root, _, files in os.walk(
            '/home/user1/projects/hubochao/datasets/benign/'):
        for f in files:
            path_list[os.path.splitext(f)[0]] = os.path.join(root, f)
    for root, _, files in os.walk(
            '/home/user1/projects/hubochao/datasets/malware/'):
        for f in files:
            path_list[os.path.splitext(f)[0]] = os.path.join(root, f)

    for i in range(len(dataset)):
        dataset[i]['path'] = path_list[dataset[i]['name']]

    with open('/home/user1/projects/hubochao/datasets/my/dataset_all.json',
              'w') as f:
        json.dump(dataset, f)


def updateDataset(
    save_path='/home/user1/projects/hubochao/datasets/my/dataset.json',
    benign_dir='/home/user1/projects/hubochao/datasets/my/benign',
    malware_dir='/home/user1/projects/hubochao/datasets/my/malware',
):
    types = {'entropy', 'stringids', 'fieldids', 'data', 'header'}

    def get_apk_features_data(path, isMal=None):
        ext = ['.npy', '.png']
        apk_list = []
        for root, _, file_names in os.walk(path):
            feature_types = set([
                os.path.splitext(file_name)[0].split('_')[-1]
                for file_name in file_names
                if os.path.splitext(file_name)[1] in ext
            ])
            if not types.issubset(feature_types):
                continue
            #满足条件
            ip_path = os.path.join(root, 'intent_permission.npy')
            if not os.path.exists(ip_path):
                ip_path = None
            apk = {
                'name': os.path.split(root)[1],
                'dex': {
                    'src': {}
                },
                'intent_permission': ip_path,
                'is_mal': isMal
            }
            flag = False
            contain_classes = False
            for file_name in file_names:
                if os.path.splitext(file_name)[1] not in ext:
                    continue
                file_base_name, img_type = os.path.splitext(
                    file_name)[0].rsplit('_', 1)
                if img_type not in types:
                    continue
                contain_classes = contain_classes or file_base_name.lower(
                ) == 'classes'
                flag = True
                if img_type not in apk['dex']:
                    apk['dex'][img_type] = {}
                apk['dex'][img_type][file_base_name] = os.path.join(
                    root, file_name)
                apk['dex']['src'][file_base_name] = os.path.join(
                    root, file_base_name + '.dex')
            if 'classes' in apk['dex']['src']:
                apk_list.append(apk)
        return apk_list

    benign_list = get_apk_features_data(benign_dir, isMal=False)
    mal_list = get_apk_features_data(malware_dir, isMal=True)
    meta_data = benign_list + mal_list
    random.shuffle(meta_data)

    with open(save_path, 'w') as f:
        json.dump(meta_data, f)
    print(
        f'saved benign sample: {len(benign_list)}, malware sample: {len(mal_list)}'
    )


#generateImage('test', target_dir, getMarkovMatrix, thread_num=1)


def preprocess():
    files = getAllFile(source_dir, ext=None)
    threading_pool = []

    class Process(multiprocessing.Process):
        def __init__(self, files, threadId):
            multiprocessing.Process.__init__(self)
            self.files = files
            self.threadId = threadId
            self.daemon = True

        def run(self):
            i = 0
            for file in self.files:
                #print('current:', file)
                extract(file)
                if True and i % 5 == 0:
                    print(
                        f'\n\033[1;32m Thread {self.threadId}: {"{:.3f}".format((i+1)/len(self.files)*100)}% completed \033[0m\n'
                    )
                i += 1

    for i in range(thread_num):
        threading_pool.append(
            Process(
                files[math.floor(i / thread_num *
                                 len(files)):math.floor((i + 1) / thread_num *
                                                        len(files))], i))
        threading_pool[-1].start()
        time.sleep(0.03)

    for thread in threading_pool:
        thread.join()


if __name__ == '__main__':
    #preprocess()
    #writeParseHeader('/home/user1/projects/hubochao/datasets/my/malware/')
    #writeParseHeader('/home/user1/projects/hubochao/datasets/my/benign/')
    '''generateImage('data',
                  '/home/user1/projects/hubochao/datasets/my/',
                  getDexSplit,
                  thread_num=24,
                  params={'type': 'data'})'''
    '''
    generateNumpy('header',
                  '/home/user1/projects/hubochao/datasets/my/',
                  getDexHeaderMatrix,
                  thread_num=40)
    '''
    '''
    generateDict(
        '/home/user1/projects/hubochao/datasets/my/malware',
        '/home/user1/projects/hubochao/datasets/my/dict_malware.json')'''
    '''updateDictionary('/home/user1/projects/hubochao/datasets/my/dict_src.json',
                     '/home/user1/projects/hubochao/datasets/my/dict.json', 10)'''
    '''generateBOW('/home/user1/projects/hubochao/datasets/my/',
                '/home/user1/projects/hubochao/datasets/my/dict.json')'''
    '''updateDataset(
        save_path='/home/user1/projects/hubochao/datasets/my/dataset_all.json')'''
    updatePath()
