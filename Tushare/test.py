import tushare as ts

ts.set_token('1adbc655bb19385f030a7a5d36bbd6a3b47a52e3a407394cc47903bf')
pro = ts.pro_api()
data = pro.stock_basic(exchange='', list_status='L',
                       fields='ts_code,symbol,name,area,industry,market,exchange,list_date')
# 涨跌幅大于等于0.2 股票最高价小于4元的 主板
rs = []
for row in data.itertuples():
    tc=row.ts_code
    if row.market == "主板":
        df = pro.query('daily', ts_code=tc, start_date='20230207', end_date='20230207')
        for row1 in df.itertuples():
            if row1.high < 4 and row1.change >= 0.2:
                rs.append(row1.symbol)

