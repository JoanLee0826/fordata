import requests
import threading, re, time, os
from queue import Queue
from lxml import etree


class MeiTu:

    proxies = {
        "http": "218.93.71.209:9999",
    }
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/"
                      "537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
    }

    def __init__(self):
        self.fir_que = Queue()  # 传递页面列表数据
        self.sec_que = Queue()  # 传递图片url数据

    def get_url(self, url):
        res = requests.get(url, headers=self.headers, proxies=self.proxies)
        html = res.content
        con = etree.HTML(html)
        pagesum = con.xpath("//div[@id='pages']/a/@href")[-2]  # 找到最后一页url
        num = re.search(r"www.meituri.com/a/(\d+)/(\d+).html", pagesum).group(1)
        page = re.search(r'www.meituri.com/a/(\d+)/(\d+).html', pagesum).group(2)
        # print(num, page)
        pic_url = r'https://www.meituri.com/a/' + num + "/"
        self.fir_que.put(pic_url)
        for each in range(1, int(page) + 1):
            pic_url = r'https://www.meituri.com/a/' + num + "/" + str(each) + ".html"
            # print(pic_url)
            self.fir_que.put(pic_url)

    def get_pic(self, url):
        res = requests.get(url, headers=self.headers, proxies=self.proxies)
        html = res.content
        con = etree.HTML(html)
        urlist = con.xpath("//div[@class='content']/img/@src")
        for url in urlist:
            self.sec_que.put(url)

    def get_db(self, url):
        g = re.search(r'.com/([^/]*)/(\d+)/(\d+)/(\d+)', url)
        num = g.group(3)
        page = g.group(4)
        res = requests.get(url, headers=self.headers, proxies=self.proxies)
        html = res.content
        path = "image/" + str(num)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + "/" + page + '.jpg', "wb") as f:
            f.write(html)

    def run(self, url):
        zero_t = threading.Thread(target=self.get_url, args=(url,))
        zero_t.start()
        time.sleep(3)
        while True:
            if not self.fir_que.empty():
                url = self.fir_que.get()
                fir_t = threading.Thread(target=self.get_pic, args=(url,))
                print("页面列表剩余--", self.fir_que.qsize())
                fir_t.start()
                time.sleep(2)
            if not self.sec_que.empty():
                url = self.sec_que.get()
                sec_t = threading.Thread(target=self.get_db, args=(url,))
                print(sec_t.name, "图片列表剩余--", self.sec_que.qsize())
                sec_t.start()
                time.sleep(0.5)
            if self.fir_que.empty() and self.sec_que.empty():
                break


meitu = MeiTu()
url = "https://www.meituri.com/a/25370/"
meitu.run(url)
time.sleep(1)
