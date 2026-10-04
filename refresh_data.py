"""Explicit, cached source collection. Set TUSHARE_TOKEN; never called on import."""
from io import StringIO
import json
import os
from pathlib import Path

import pandas as pd
import requests
import tushare as ts

from allocation import ASSETS, ROOT


def main():
    folder = ROOT / 'data' / 'validated_20260930'
    folder.mkdir(parents=True, exist_ok=True)
    pro = ts.pro_api(os.environ['TUSHARE_TOKEN'], timeout=30)
    calendar_path = folder / 'trade_cal.csv'
    if not calendar_path.exists():
        calendar = pro.trade_cal(exchange='SSE', start_date='20180101', end_date='20260930')
        if calendar.empty or len(calendar) >= 6000:
            raise ValueError('Invalid exchange calendar response')
        calendar.to_csv(calendar_path, index=False)
    panels = []
    for symbol in ASSETS:
        sources = {}
        for endpoint in ['fund_daily', 'fund_adj']:
            path = folder / f'{endpoint}_{symbol}.csv'
            if not path.exists():
                frame = pro.query(endpoint, ts_code=symbol, start_date='20180101', end_date='20260930')
                if frame.empty or len(frame) >= 2000:
                    # fund_daily is capped at 2000 on some provider plans. Fetch calendar years.
                    parts = []
                    for year in range(2018, 2027):
                        part = pro.query(endpoint, ts_code=symbol, start_date=f'{year}0101',
                                         end_date=min(f'{year}1231', '20260930'))
                        if part.empty or len(part) >= 2000:
                            raise ValueError(f'Incomplete {endpoint}: {symbol} {year}')
                        parts.append(part)
                    frame = pd.concat(parts, ignore_index=True)
                frame.to_csv(path, index=False)
            sources[endpoint] = pd.read_csv(path, dtype={'trade_date': str})
        daily, factors = sources['fund_daily'], sources['fund_adj']
        merged = daily.merge(factors[['trade_date', 'adj_factor']], on='trade_date',
                             validate='one_to_one', how='left').sort_values('trade_date')
        if merged.adj_factor.isna().any() or merged.empty:
            raise ValueError(f'Missing adjustment factors: {symbol}')
        scale = merged.adj_factor / merged.adj_factor.iloc[-1]
        for col in ['open', 'high', 'low', 'close']:
            merged[col] *= scale
        merged['date'] = pd.to_datetime(merged.trade_date)
        merged['symbol'] = symbol
        panels.append(merged[['date','symbol','open','high','low','close','vol','amount']])
        print(symbol, len(merged), flush=True)
    panel = pd.concat(panels).sort_values(['date','symbol'])
    panel.to_csv(folder / 'etf_daily.csv', index=False)
    path = folder / 'index_dailybasic.csv'
    if not path.exists():
        pe = pro.index_dailybasic(ts_code='000300.SH', start_date='20180101', end_date='20260930',
                                  fields='ts_code,trade_date,pe_ttm')
        if pe.empty or len(pe) >= 3000:
            raise ValueError('Empty or capped valuation response')
        pe.to_csv(path, index=False)
    for year in range(2018, 2027):
        path = folder / f'chinabond_{year}.csv'
        if path.exists():
            continue
        response = requests.get('https://yield.chinabond.com.cn/cbweb-pbc-web/pbc/historyQuery',
            params={'startDate': f'{year}-01-01', 'endDate': min(f'{year}-12-31','2026-09-30'),
                    'gjqx':'0','qxId':'ycqx','locale':'cn_ZH'},
            headers={'User-Agent':'Mozilla/5.0'}, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text.replace('&nbsp','')), header=0)
        curve = next(t for t in tables if '10年' in t.columns and '日期' in t.columns)
        if curve.empty:
            raise ValueError(f'Empty yield response {year}')
        curve.to_csv(path, index=False)
    manifest = {'retrieved_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'cutoff': '2026-09-30', 'prices': 'Tushare fund_daily + fund_adj; qfq OHLC, endpoint raw snapshots retained',
        'price_rows': len(panel), 'currency': 'CNY', 'valuation': 'Tushare index_dailybasic, 000300.SH pe_ttm',
        'yield': 'ChinaBond official historyQuery, 中债国债收益率曲线, 10年, percent divided by 100',
        'sources': ['https://tushare.pro/document/2?doc_id=127', 'https://tushare.pro/document/2?doc_id=199',
                    'https://tushare.pro/document/2?doc_id=128',
                    'https://yield.chinabond.com.cn/cbweb-pbc-web/pbc/historyQuery'],
        'limitations': ['Historical vintage downloaded today; not PIT/revision certified',
                       'Adjusted ETF price is reinvested-distribution proxy, not fund NAV',
                       'No documented historical trade-size/liquidity/slippage model',
                       'Tushare yc_cb permission unavailable; official ChinaBond source used explicitly']}
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
