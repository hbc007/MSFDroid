import xmltodict, json, re
'''
def xml_to_json(xml_str):
    # parse是的xml解析器
    xml_parse = xmltodict.parse(xml_str)
    # json库dumps()是将dict转化成json格式,loads()是将json转化成dict格式。
    # dumps()方法的ident=1,格式化json
    json_str = json.dumps(xml_parse, indent=1)
    print(json_str)
    return xml_parse


def read(path):
    with open(path, 'r') as f:
        xml = ''.join(f.readlines())
        data = xml_to_json(xml)
        return data
        permissions = [
            x['@android:name'] for x in data['manifest']['uses-permission']
        ]
        print(permissions)


data=read(
    '/home/user1/projects/hubochao/datasets/my/malware/CIC2019/0a8a0bf71a8b3196d5a1ec1144ec15e6/AndroidManifest.xml'
)
permissions = [x['@android:name'] for x in data['manifest']['uses-permission']]
permissions = [x['@android:name'] for x in data['manifest']['application']]
data
'''
'''
def read(path):
    with open(path, 'r') as f:
        xml = ''.join(f.readlines())
        key = re.findall(r'android:name=\"(.*)\"', xml)
        val = [1] * len(key)
        print(dict(zip(key, val)))


read(
    '/home/user1/projects/hubochao/datasets/my/malware/CIC2019/0a8a0bf71a8b3196d5a1ec1144ec15e6/AndroidManifest.xml'
)'''
dic = {}
with open('/home/user1/projects/hubochao/datasets/my/dict_src.json', 'r') as f:
    dic = json.load(f)
    a = [(x, dic[x]) for x in dic
         if x.find('android.intent') == 0 or x.find('android.permission') == 0]
    print(a)
