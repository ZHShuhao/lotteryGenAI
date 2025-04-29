# import http.client
# import json
#
# conn = http.client.HTTPSConnection("mega-millions.p.rapidapi.com")
#
# headers = {
#     'x-rapidapi-key': "b5b78de802msh45b4a624c9a36b0p196f8djsn0b3ec34ef836",
#     'x-rapidapi-host': "mega-millions.p.rapidapi.com"
# }
#
# conn.request("GET", "/", headers=headers)
#
# res = conn.getresponse()
# data = res.read()
#
# decoded_data = data.decode("utf-8")  # 解码为字符串
#
# # 解析 JSON 数据
# json_data = json.loads(decoded_data)  # 转换为 Python 字典
#
# print(json.dumps(json_data, indent=4))
#
#
# # 提取 `data` 列表中的字段
# draws = json_data["data"]
# csv_data = [["DrawingDate", "Number1", "Number2", "Number3", "Number4", "Number5", "MegaBall", "Megaplier", "JackPot"]]

import http.client
import json

def fetch_mega_millions_data():
    conn = http.client.HTTPSConnection("mega-millions.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "b5b78de802msh45b4a624c9a36b0p196f8djsn0b3ec34ef836",
        'x-rapidapi-host': "mega-millions.p.rapidapi.com"
    }
    conn.request("GET", "/", headers=headers)
    res = conn.getresponse()
    data = res.read()

    # Parse JSON and return the desired fields
    json_data = json.loads(data.decode("utf-8"))
    draws = json_data["data"]
    extracted_data = [
        {
            "DrawingDate": draw["DrawingDate"],
            "Number1": draw["FirstNumber"],
            "Number2": draw["SecondNumber"],
            "Number3": draw["ThirdNumber"],
            "Number4": draw["FourthNumber"],
            "Number5": draw["FifthNumber"],
            "MegaBall": draw["MegaBall"],
            "Megaplier": draw["Megaplier"],
            "JackPot": draw["JackPot"]
        }
        for draw in draws
    ]
    return extracted_data
