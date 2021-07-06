def get_before_after_trade_days(date, count):
    """
    date :查询日期
    count : 前后追朔的数量
    is_before : True , 前count个交易日  ; False ,后count个交易日

    返回 : 基于date的日期, 向前或者向后count个交易日的日期 ,一个datetime.date 对象
    """
    from jqdata import  get_trade_days
    import pandas as pd
    all_date = pd.Series(get_all_trade_days())
    if isinstance(date,str):
        all_date = all_date.astype(str)
    if isinstance(date,datetime.datetime):
        date = date.date()

    all_date[all_date>date].head(count).values[-1]

    return all_date



date = "2019-10-20"#datetime.date(2019,10,20)
count = 10
all_date = get_before_after_trade_days(date, count)
print(all_date)
