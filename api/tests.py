from django.test import TestCase

# Create your tests here.
import requests

def test():
    url = "http://127.0.0.1:8000/api/get_label"
    data = {"text": "我抽着差不多的烟,又过了差不多的一天！"}
    # 使用requests发送post请求
    res = requests.post(url, data=data)
    print(res.text)

if __name__ == "__main__":
    test()
    print(__file__)