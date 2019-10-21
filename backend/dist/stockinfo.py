import pandas as pd


pd.set_option('display.max_rows', 500)
# print(pd.set_option.key())
code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0]

# 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌

code_df.종목코드 = code_df.종목코드.map('{:06d}'.format)

code_df = code_df[['회사명', '종목코드']]

code_df = code_df.rename(columns={'회사명': 'name', '종목코드': 'code'})

print(code_df.head())

def get_url(item_name, code_df):
    code = code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)
    print(code[1:])
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code[1:])

    print("요청 URL = {}".format(url))

    return url

item_name ='삼성전자'
url = get_url(item_name, code_df)

df = pd.DataFrame()

for page in range(1, 5):
    pg_url = '{url}&page={page}'.format(url=url, page=page)
    df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)
    # print(df)
df = df.dropna()

df.drop(['날짜', '전일비'], axis=1, inplace=True)

dataset_temp = df.as_matrix()


print(df)