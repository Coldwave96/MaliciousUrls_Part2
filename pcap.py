# coding:utf-8

try:
    import scapy.all as scapy
except ImportError:
    import scapy

try:
    # This import works from the project directory
    import scapy_http.http as http
except ImportError:
    # If you installed this package via pip, you just need to execute this
    from scapy.layers import http


pkts = scapy.rdpcap('test.pcap')
req_list = []


def get_url(pkt):
    if not pkt.haslayer(http.HTTPRequest):
        # This packet doesn't contain an HTTP request so we skip it
        return
    http_layer = pkt[http.HTTPRequest].fields
    req_url = http_layer['Path'].decode() + '\n'
    req_list.append(req_url)


for packet in pkts:
    get_url(packet)

f = open('data/train/poc_test.txt', 'w')
f.writelines(req_list)
f.close()
