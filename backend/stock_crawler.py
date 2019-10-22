import requests
import json

from bs4 import BeautifulSoup


def get_business_types():
    return requests.get("http://localhost:8000/stock/businessTypes/").json()


def get_corparations(business_code):
    return requests.get("http://localhost:8000/stock/businessTypes/"+business_code).json()


def preproces_str_to_int(str):
    return int(str.strip().replace(',', ''))


if __name__ == '__main__':
    # business_types = get_business_types()

    corparations = [{"name": "삼성전자", "corp_code": "005930"}]
    # for business_type in business_types:
    #     corparations += get_corparations(business_type['business_code'])

    for corp in corparations:
        res = requests.get("https://finance.naver.com/item/sise_day.nhn?code=" + corp['corp_code'])

        soup = BeautifulSoup(res.text, 'html.parser')
        last_page_element = soup.select('body > table.Nnavi > tr > td.pgRR > a')
        last_pagenum = last_page_element[0].get("href").split("page=")[1]

        for i in reversed(range(1, int(last_pagenum))):
            page_res = requests.get("https://finance.naver.com/item/sise_day.nhn?code={}&page={}"
                                    .format(corp['corp_code'], i))

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
                        volumn = preproces_str_to_int(row[6].text)
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
                                    "volumn": volumn
                                }
                            ]
                        }
                        requests.post("http://localhost:8000/stock/" + corp['corp_code'] + "/", data=json.loads(json.dumps(data)))
