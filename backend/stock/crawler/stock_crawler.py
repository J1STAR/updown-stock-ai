import requests
from multiprocessing import Pool
from bs4 import BeautifulSoup
import time
import os


def get_business_types():
    return requests.get("http://localhost:8000/stock/businessTypes/").json()


def get_corparations(business_code):
    return requests.get("http://localhost:8000/stock/businessTypes/"+business_code).json()


def preproces_str_to_int(str):
    return int(str.strip().replace(',', ''))


def crawl_corp_stock_info(corp):
    stock_data_list = []

    res = requests.get("https://finance.naver.com/item/sise_day.nhn?code=" + corp['corp_code'])

    soup = BeautifulSoup(res.text, 'html.parser')
    last_page_element = soup.select('body > table.Nnavi > tr > td.pgRR > a')

    if len(last_page_element) == 0:
        last_pagenum = 1
    else:
        last_pagenum = last_page_element[0].get("href").split("page=")[1]
    print("pid : {} / CODE: {} / NAME: {} / PAGE_NUM: {}".format(os.getpid(), corp['corp_code'], corp['name'], last_pagenum))

    for i in reversed(range(int(last_pagenum))):
        try:
            page_res = requests.get("https://finance.naver.com/item/sise_day.nhn?code={}&page={}"
                                    .format(corp['corp_code'], i+1))

            soup = BeautifulSoup(page_res.text, 'html.parser')
            stock_table_tr = soup.select('table tr')

            for tr in reversed(stock_table_tr):
                if len(tr.attrs) is not 0:
                    row = tr.find_all('span')
                    if len(row) is not 0:
                        date = row[0].text
                        closing_price = preproces_str_to_int(row[1].text)

                        diff = preproces_str_to_int(row[2].text)
                        if any("nv" in c for c in row[2].get('class')):
                            diff *= -1

                        open_price = preproces_str_to_int(row[3].text)
                        high_price = preproces_str_to_int(row[4].text)
                        low_price = preproces_str_to_int(row[5].text)
                        volume = preproces_str_to_int(row[6].text)
                        data = {
                            "corp_name": corp['name'],
                            "stock_info": [
                                {
                                    "date": date,
                                    "closing_price": closing_price,
                                    "diff": diff,
                                    "open_price": open_price,
                                    "high_price": high_price,
                                    "low_price": low_price,
                                    "volume": volume
                                },
                            ]
                        }
                        stock_data_list.append(data)
        except:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            print("ZZzzzz...")
            time.sleep(5)
            print("Was a nice sleep, now let me continue...")
            continue

    # print(stock_data_list)
    requests.post("http://localhost:8000/stock/corp/" + corp['corp_code'] + "/", json=stock_data_list)


def crawl_corp_stock_info_1day(corp):
    res = requests.get("https://finance.naver.com/item/sise_day.nhn?code=" + corp['corp_code'])

    soup = BeautifulSoup(res.text, 'html.parser')
    last_page_element = soup.select('body > table.Nnavi > tr > td.pgRR > a')

    print("pid : {} / CODE: {} / NAME: {} / PAGE_NUM: {}".format(os.getpid(), corp['corp_code'], corp['name'], 1))

    try:
        page_res = requests.get("https://finance.naver.com/item/sise_day.nhn?code={}&page=1"
                                .format(corp['corp_code']))

        soup = BeautifulSoup(page_res.text, 'html.parser')
        stock_table_tr = soup.select('table tr:nth-child(3)')

        data = {}
        if len(stock_table_tr[0].attrs) is not 0:
            row = stock_table_tr[0].find_all('span')
            if len(row) is not 0:
                date = row[0].text
                closing_price = preproces_str_to_int(row[1].text)

                diff = preproces_str_to_int(row[2].text)
                if any("nv" in c for c in row[2].get('class')):
                    diff *= -1

                open_price = preproces_str_to_int(row[3].text)
                high_price = preproces_str_to_int(row[4].text)
                low_price = preproces_str_to_int(row[5].text)
                volume = preproces_str_to_int(row[6].text)
                data = {
                    "corp_name": corp['name'],
                    "stock_info": [
                        {
                            "date": date,
                            "closing_price": closing_price,
                            "diff": diff,
                            "open_price": open_price,
                            "high_price": high_price,
                            "low_price": low_price,
                            "volume": volume
                        },
                    ]
                }
    except:
        print("Connection refused by the server..")
        print("Let me sleep for 5 seconds")
        print("ZZzzzz...")
        time.sleep(5)
        print("Was a nice sleep, now let me continue...")

    # print(data)
    requests.post("http://localhost:8000/stock/corp/" + corp['corp_code'] + "/", json=data)


if __name__ == '__main__':
    business_types = get_business_types()

    corparations = []
    for business_type in business_types:
        corparations += get_corparations(business_type['business_code'])

    crawl_processes = []
    print("Corp size > ", len(corparations))

    pool = Pool(processes=4)
    pool.map(crawl_corp_stock_info_1day, corparations)
