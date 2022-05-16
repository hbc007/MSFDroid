from struct import unpack

total_chunk_size = 0x00
string_pool = []
xml_resource_map_list = []
namespace = {}
tab = 0
element_count = 0


class HEADER_DATA:
    def __init__(self, Header_Type, Header_Size, Chunk_Size):
        self.Header_Type = Header_Type
        self.Header_Size = Header_Size
        self.Chunk_Size = Chunk_Size

    def getHeaderType(self):
        return self.Header_Type

    def getHeaderSize(self):
        return self.Header_Size

    def getChunkSize(self):
        return self.Chunk_Size

    def print(self):
        print(hex(self.Header_Type), "\t" + "HEADER_TYPE")
        print(hex(self.Header_Size), "\t" + "HEADER_SIZE")
        print(hex(self.Chunk_Size), "\t" + "CHUNK_SIZE")

    def getBodySize(self):
        return self.Chunk_Size - 8


#STRING_POOL DATA뽑기
def getStringBodyData(STRING_POOL_BODY):
    stringCount = int(unpack("<L", STRING_POOL_BODY[0:4])[0])
    styleCount = int(unpack("<L", STRING_POOL_BODY[4:8])[0])
    flags = unpack("<L", STRING_POOL_BODY[8:12])[0]
    stringsStart = unpack("<L", STRING_POOL_BODY[12:16])[0]  #address
    stylesStart = unpack("<L", STRING_POOL_BODY[16:20])[0]  #address
    # print(hex(stringCount),"\t"+"STRING_COUNT")
    # print(hex(styleCount),"\t"+"STYLE_COUNT")
    # print(hex(flags),"\t"+"FLAGS")
    # print(hex(stringsStart),"\t"+"STRINGS_START")
    # print(hex(stylesStart),"\t"+"STYLES_START")
    x = 20  #read 위치 시작
    y = x + 4  # read 위치 끝

    strings_offsets = []
    strings = []
    for i in range(0, stringCount):
        strings_offsets.append(int(unpack("<L", STRING_POOL_BODY[x:y])[0]))
        x = y
        y += 4

    y -= 2  # string은 두글자씩 읽으므로
    for i in range(0, stringCount):
        string = ''
        if i == stringCount - 1:  #마지막 string 인 경우
            length = unpack("<H", STRING_POOL_BODY[x:y])[0]
            x += 2
            last_string = STRING_POOL_BODY[x:(x + length) * 2]

            for i in range(0, length):
                data = unpack("<H",
                              last_string[2 * i:2 * (i + 1)])[0]  #마지막 스트링 더하기
                string += chr(data)
            strings.append(string)

        else:
            string_start = strings_offsets[i]
            string_end = strings_offsets[i + 1]
            length = unpack("<H", STRING_POOL_BODY[x:y])[0]
            count = 0
            x = y
            y += 2
            for z in range(string_start + 2, string_end, 2):
                if count == length:
                    #offset 건너띄기
                    x = y
                    y += 2
                    continue
                data = unpack("<H", STRING_POOL_BODY[x:y])[0]
                x = y
                y += 2
                string += chr(data)
                count += 1

            strings.append(string)
    # print("--string list--")
    #print(strings)
    return strings


#XML_RESOURCE_MAP DATA 뽑기
def getResourceMapBodyData(XML_RESOURCE_MAP_BODY, size):
    xml_resource_map_list = []
    for i in range(int(size / 4)):
        data = unpack("<L", XML_RESOURCE_MAP_BODY[i * 4:(i + 1) * 4])[0]
        xml_resource_map_list.append(data)
    # print(xml_resource_map_list)
    return xml_resource_map_list  #address list


#XML_START_NAMESPACE DATA 뽑기
def getStartNameSpaceData(XML_START_NAMESPACE_BODY):
    line_number = unpack("<L", XML_START_NAMESPACE_BODY[0:4])[0]
    comment = unpack("<L", XML_START_NAMESPACE_BODY[4:8])[0]
    prefix = unpack("<L", XML_START_NAMESPACE_BODY[8:12])[0]
    uri = unpack("<L", XML_START_NAMESPACE_BODY[12:16])[0]
    namespace[string_pool[uri]] = string_pool[prefix]


#XML_END_NAME_SPACE DATA 뽑기
def getEndNameSpaceData(XML_END_NAMESPACE_BODY):
    line_number = unpack("<L", XML_END_NAMESPACE_BODY[0:4])[0]
    comment = unpack("<L", XML_END_NAMESPACE_BODY[4:8])[0]
    prefix = unpack("<L", XML_END_NAMESPACE_BODY[8:12])[0]
    uri = unpack("<L", XML_END_NAMESPACE_BODY[12:16])[0]


#XML_START_ELEMENT DATA 뽑기
def getStartElementData(XML_START_ELEMENT_BODY):
    #-----element의 정보 알려줌
    line_number = unpack("<L", XML_START_ELEMENT_BODY[0:4])[0]
    comment = unpack("<L", XML_START_ELEMENT_BODY[4:8])[0]
    ns = unpack("<L", XML_START_ELEMENT_BODY[8:12])[0]
    name = unpack("<L", XML_START_ELEMENT_BODY[12:16])[0]  # LinearLayout
    attribute_start = unpack("<H", XML_START_ELEMENT_BODY[16:18])[0]
    attribute_size = unpack("<H", XML_START_ELEMENT_BODY[18:20])[0]
    attribute_count = unpack("<H", XML_START_ELEMENT_BODY[20:22])[0]
    id_index = unpack("<H", XML_START_ELEMENT_BODY[22:24])[0]
    class_index = unpack("<H", XML_START_ELEMENT_BODY[24:26])[0]
    style_index = unpack("<H", XML_START_ELEMENT_BODY[26:28])[0]

    #print 요소 추가
    print_element = ((tab - 1) * "\t" + "<")
    print_element += string_pool[name] + "\n" + tab * "\t"

    if element_count == 0:
        for string in string_pool:
            if "http://schemas.android.com/apk/res/android" == string:
                print_element += ' xmlns:android="' + string + '"\n' + tab * "\t"
                break

    attributes = []
    for i in range(attribute_count):
        #attribute는 body 28지점부터 시작 , attribute size는 고정
        start = 28 + i * attribute_size
        end = start + attribute_size
        attribute = XML_START_ELEMENT_BODY[start:end]
        attribute_ns = unpack("<L",
                              attribute[0:4])[0]  #사용될 문자열 string_pool index
        attribute_name = unpack("<L", attribute[4:8])[0]  #사용될 문자열 index
        attribute_raw_value = unpack("<L", attribute[8:12])[
            0]  #원리 XML에 포함된 XML chunk의 string_pool의 index로 나타내는 속성의 값 지정
        attribute_data_size = unpack("<H", attribute[12:14])[0]
        #attribute[14] == 0
        attribute_data_type = attribute[15]
        attribute_data = unpack("<L", attribute[16:20])[0]
        print_element += " "

        if attribute_ns == 0xffffffff:
            continue  # element has no namespace name
        #---attribute print
        print_element += (namespace[string_pool[attribute_ns]] + ":")
        print_element += string_pool[attribute_name]

        #data type에 따라 로직 처리
        if attribute_data_type == 0x01:  #resource
            print_element += ('="@' + hex(attribute_data) + '"')
        elif attribute_data_type == 0x03:  #string값 -> string_pool에서 가져옴
            print_element += ('="' + string_pool[attribute_raw_value] + '"')
        elif attribute_data_type == 0x10:  #int값
            print_element += ('="' + str(attribute_data) + '"')
        elif attribute_data_type == 0x11:  #hex값
            print_element += ('="' + hex(attribute_data) + '"')
        elif attribute_data_type == 0x12:  #boolean 값
            print_element += ('="' + str(bool(attribute_data)) + '"')
        print_element += '\n' + tab * '\t'
        attributes.append(attribute)
    if print_element[len(print_element) - 1] == ' ':
        print_element = print_element[:len(print_element) - 2]
    print_element += ">"
    #line 출력
    #print(print_element)
    return print_element


#XML_END_ELEMENT DATA 뽑기
def getEndElementData(XML_END_ELEMENT_BODY):
    line_number = unpack("<L", XML_END_ELEMENT_BODY[0:4])[0]
    comment = unpack("<L", XML_END_ELEMENT_BODY[4:8])[0]
    ns = unpack("<L", XML_END_ELEMENT_BODY[8:12])[0]
    name = unpack("<L", XML_END_ELEMENT_BODY[12:16])[0]

    print_element = (tab - 1) * "\t" + "</" + string_pool[name] + ">"
    #print(print_element)
    return print_element


#file = open("/home/user1/projects/hubochao/datasets/my/malware/0025f94ed7defb1f063b30463612f6795c0c5aa0786d51f49722e17962d2bae2/AndroidManifest.xml","rb")
def decode(file):
    global total_chunk_size, string_pool, xml_resource_map_list, namespace, tab, element_count
    decode_str = ''
    total_chunk_size = 0x00
    string_pool = []
    xml_resource_map_list = []
    namespace = {}
    tab = 0
    element_count = 0
    HEADER = HEADER_DATA(
        unpack("<H", file.read(2))[0],
        unpack("<H", file.read(2))[0],
        unpack("<L", file.read(4))[0])
    total_chunk_size += HEADER.getHeaderSize()

    STRING_POOL_CHUNK = []
    STRING_POOL_HEADER = HEADER_DATA(
        unpack("<H", file.read(2))[0],
        unpack("<H", file.read(2))[0],
        unpack("<L", file.read(4))[0])
    STRING_POOL_CHUNK.append(STRING_POOL_HEADER)
    STRING_POOL_BODY = file.read(STRING_POOL_HEADER.getBodySize())
    STRING_POOL_CHUNK.append(STRING_POOL_BODY)

    string_pool = getStringBodyData(STRING_POOL_BODY)
    total_chunk_size += STRING_POOL_HEADER.getChunkSize()

    XML_RESOURCE_MAP_CHUNK = []
    XML_RESOURCE_MAP_HEADER = HEADER_DATA(
        unpack("<H", file.read(2))[0],
        unpack("<H", file.read(2))[0],
        unpack("<L", file.read(4))[0])
    XML_RESOURCE_MAP_CHUNK.append(XML_RESOURCE_MAP_HEADER)
    XML_RESOURCE_MAP_BODY = file.read(XML_RESOURCE_MAP_HEADER.getBodySize())
    XML_RESOURCE_MAP_CHUNK.append(XML_RESOURCE_MAP_BODY)

    xml_resource_map_list = getResourceMapBodyData(
        XML_RESOURCE_MAP_BODY, XML_RESOURCE_MAP_HEADER.getBodySize())
    total_chunk_size += XML_RESOURCE_MAP_HEADER.getChunkSize()

    decode_str += '<?xml version="1.0" encoding="utf-8"?>\n'
    while True:
        if total_chunk_size == HEADER.Chunk_Size: break
        XML_CHUNK = []
        XML_HEADER = HEADER_DATA(
            unpack("<H", file.read(2))[0],
            unpack("<H", file.read(2))[0],
            unpack("<L", file.read(4))[0])
        XML_CHUNK.append(XML_HEADER)
        XML_BODY = file.read(XML_HEADER.getBodySize())
        data = None
        if (XML_HEADER.getHeaderType() == 0x0100):
            getStartNameSpaceData(XML_BODY)
        elif (XML_HEADER.getHeaderType() == 0x0101):
            getEndNameSpaceData(XML_BODY)
        elif (XML_HEADER.getHeaderType() == 0x0102):
            tab += 1
            data = getStartElementData(XML_BODY)
            element_count += 1
        elif (XML_HEADER.getHeaderType() == 0x0103):
            data = getEndElementData(XML_BODY)
            tab -= 1
            element_count -= 1
        if data:
            decode_str += data + '\n'
        total_chunk_size += XML_HEADER.getChunkSize()
    return decode_str
