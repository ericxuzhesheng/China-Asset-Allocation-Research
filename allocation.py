"""Causal CNY ETF allocation with explicit next-open self-financing accounting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = ['510300.SH', '511010.SH', '518880.SH']
ROOT = Path(__file__).resolve().parent


def load_prices(path):
    data = pd.read_csv(path, parse_dates=['date'])
    data = data.loc[data.symbol.isin(ASSETS)].copy()
    if data.duplicated(['date', 'symbol']).any():
        raise ValueError('Duplicate asset/date prices')
    if {'high', 'low'}.issubset(data.columns):
        if ((data.high < data[['open','close','low']].max(axis=1)) |
                (data.low > data[['open','close','high']].min(axis=1))).any():
            raise ValueError('Impossible OHLC prices')
    prices = data.pivot(index='date', columns='symbol', values=['open', 'close']).sort_index()
    prices = prices.reindex(columns=pd.MultiIndex.from_product([['open', 'close'], ASSETS]))
    if prices.empty or prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError('Prices must cover all three assets on every session; no forward filling')
    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError('Nonfinite prices')
    calendar_path = Path(path).with_name('trade_cal.csv')
    if calendar_path.exists():
        calendar = pd.read_csv(calendar_path, dtype={'cal_date': str})
        expected = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open.eq(1), 'cal_date'])).sort_values()
        expected = expected[(expected >= prices.index[0]) & (expected <= prices.index[-1])]
        if not prices.index.equals(expected.rename('date')):
            raise ValueError('Price sessions differ from the exchange calendar')
    return prices


def load_erp(folder):
    pe = pd.read_csv(folder / 'index_dailybasic.csv', dtype={'trade_date': str})
    pe['date'] = pd.to_datetime(pe.trade_date)
    pe = pe.set_index('date').sort_index()['pe_ttm']
    curves = pd.concat([pd.read_csv(p) for p in sorted(folder.glob('chinabond_*.csv'))])
    curves = curves.loc[curves['曲线名称'].eq('中债国债收益率曲线')].copy()
    curves['date'] = pd.to_datetime(curves['日期'])
    if curves.empty or curves.date.duplicated().any() or pe.index.duplicated().any():
        raise ValueError('Missing or duplicate government yield / valuation observations')
    yields = pd.to_numeric(curves.set_index('date')['10年']).sort_index() / 100
    if pe.le(0).any() or pe.isna().any() or yields.isna().any():
        raise ValueError('Invalid PE/yield input')
    # Match only same-date observations. Signal lookup below has a seven-day expiry.
    result = pd.concat([1 / pe.rename('pe_ttm'), yields.rename('yield_10y')], axis=1, sort=True).dropna()
    result.columns = ['earnings_yield', 'yield_10y']
    result['erp'] = result.earnings_yield - result.yield_10y
    return result


def signals(prices, erp, lookback=120):
    """Month-end decisions, daily-session momentum, dated ERP strictly before decision."""
    if lookback < 1:
        raise ValueError('lookback must be a positive trading-session count')
    dates = prices.index
    end_positions = pd.Series(np.arange(len(dates)), index=dates).groupby(dates.to_period('M')).last()
    gold_momentum = prices['close'][ASSETS[2]].pct_change(lookback, fill_method=None)
    records = []
    for i in end_positions:
        if i + 1 >= len(dates) or pd.isna(gold_momentum.iloc[i]):
            continue
        eligible = erp.loc[(erp.index < dates[i]) & (erp.index >= dates[i] - pd.Timedelta(days=7))]
        if eligible.empty:
            raise ValueError(f'Missing fresh ERP at decision {dates[i].date()}')
        row = eligible.iloc[-1]
        records.append({'decision_date': dates[i], 'execution_date': dates[i + 1],
                        'erp_source_date': eligible.index[-1], 'erp': row.erp,
                        'gold_momentum_120_sessions': gold_momentum.iloc[i]})
    if not records:
        raise ValueError('Insufficient history for valid signals')
    return pd.DataFrame(records).set_index('execution_date')


def targets(decisions, threshold, gold_cap=0.10, static=False):
    if not 0 <= gold_cap <= 1 or not np.isfinite(threshold):
        raise ValueError('Invalid portfolio policy')
    equity = np.full(len(decisions), .6) if static else np.where(decisions.erp > threshold, .8, .2)
    gold = np.where(decisions.gold_momentum_120_sessions > 0, gold_cap, 0.)
    return pd.DataFrame({ASSETS[0]: equity * (1-gold), ASSETS[1]: (1-equity) * (1-gold),
                         ASSETS[2]: gold}, index=decisions.index)


def backtest(prices, schedule, cost_bps=5.):
    """Trade target NAV weights at open; costs charged on both buys and sells.

    Units drift between trades. Target dollar positions and cost are solved together,
    so costs cannot create leverage. No trade uses the same day's close.
    """
    if cost_bps < 0 or cost_bps >= 1000:
        raise ValueError('Invalid cost')
    if schedule.empty or not schedule.index.isin(prices.index).all():
        raise ValueError('Invalid execution schedule')
    if schedule.index.duplicated().any() or not schedule.index.is_monotonic_increasing:
        raise ValueError('Invalid execution order')
    if (schedule < 0).any().any() or not np.allclose(schedule.sum(axis=1), 1):
        raise ValueError('Targets must be unlevered, fully invested weights')
    prices = prices.loc[schedule.index[0]:]
    units = np.zeros(3)
    cash = nav_previous = 1.
    records = []
    for date, row in prices.iterrows():
        opening, closing = row['open'].to_numpy(), row['close'].to_numpy()
        before = units * opening
        open_nav = before.sum() + cash
        cost = turnover = 0.
        if date in schedule.index:
            weights = schedule.loc[date, ASSETS].to_numpy()
            # Contraction because the cost rate is small; solve actual traded dollars.
            net_open = open_nav
            for _ in range(60):
                cost = np.abs(net_open * weights - before).sum() * cost_bps / 10000
                net_open = open_nav - cost
            turnover = np.abs(net_open * weights - before).sum() / open_nav
            units = net_open * weights / opening
            cash = 0.
            gross_nav = open_nav * np.dot(weights, closing / opening)
        else:
            gross_nav = np.dot(units, closing) + cash
        nav = np.dot(units, closing) + cash
        record = {'date': date, 'nav': nav, 'return': nav / nav_previous - 1,
                  'gross_return': gross_nav / nav_previous - 1, 'open_nav': open_nav,
                  'cost_cash': cost, 'turnover': turnover, 'rebalance': date in schedule.index}
        for j, asset in enumerate(ASSETS):
            record[f'units_{asset}'] = units[j]
            record[f'open_{asset}'] = opening[j]
            record[f'close_{asset}'] = closing[j]
            record[f'weight_{asset}'] = units[j] * opening[j] / (open_nav-cost)
        records.append(record)
        nav_previous = nav
    return pd.DataFrame(records).set_index('date')


def metrics(returns):
    returns = returns.astype(float)
    if returns.empty or returns.isna().any():
        raise ValueError('Empty or missing evaluation returns')
    wealth = np.r_[1., np.cumprod(1 + returns.to_numpy())]
    years = len(returns) / 252
    vol = float(returns.std(ddof=1) * np.sqrt(252))
    return {'sessions': len(returns), 'total_return': float(wealth[-1]-1),
            'annual_return': float(wealth[-1] ** (1/years)-1), 'annual_volatility': vol,
            'sharpe_rf0': float(returns.mean()*252/vol) if vol > 1e-12 else None,
            'max_drawdown': float(np.min(wealth / np.maximum.accumulate(wealth)-1))}


def run(data_dir, out_dir):
    prices = load_prices(data_dir / 'etf_daily.csv')
    erp = load_erp(data_dir)
    decisions = signals(prices, erp)
    calibration = decisions.loc[decisions.index < '2023-01-01']
    if calibration.empty:
        raise ValueError('No training observations before 2023')
    # Threshold is calibrated only on training signals, never on test returns.
    threshold = float(calibration.erp.median())
    test = decisions.loc[decisions.index >= '2023-01-01']
    if test.empty:
        raise ValueError('No test decisions')
    prices_test = prices.loc[test.index[0]:]
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(out_dir / 'decisions.csv')
    erp.to_csv(out_dir / 'erp_observations.csv')
    policies = {'erp_gold': targets(test, threshold), 'erp_only': targets(test, threshold, 0),
                '6040_gold': targets(test, threshold, static=True),
                '6040': targets(test, threshold, 0, True),
                'original_zero_threshold': targets(test, 0)}
    static_8020 = targets(test, threshold)
    gold_weight = static_8020[ASSETS[2]]
    static_8020[ASSETS[0]] = .8 * (1-gold_weight)
    static_8020[ASSETS[1]] = .2 * (1-gold_weight)
    policies['8020_gold'] = static_8020
    for name, w in [('equity_buy_hold', [1.,0.,0.]), ('bond_buy_hold', [0.,1.,0.]),
                    ('gold_buy_hold', [0.,0.,1.])]:
        policies[name] = pd.DataFrame([w], index=test.index[:1], columns=ASSETS)
    summary, navs = [], {}
    for cost in [0., 5., 10.]:
        for name, policy in policies.items():
            ledger = backtest(prices_test, policy, cost)
            ledger.to_csv(out_dir / f'{name}_{int(cost)}bps.csv')
            summary.append({'strategy': name, 'cost_bps_per_side': cost, **metrics(ledger['return'])})
            if cost == 5:
                navs[name] = ledger.nav
    table = pd.DataFrame(summary)
    table.to_csv(out_dir / 'metrics.csv', index=False)
    assumptions = {'price_start': str(prices.index[0].date()), 'price_end': str(prices.index[-1].date()),
                   'calibration_end': '2022-12-31', 'calibration_signals': len(calibration),
                   'test_start': str(prices_test.index[0].date()), 'test_end': str(prices_test.index[-1].date()),
                   'threshold_training_median': threshold, 'test_decisions': len(test),
                   'test_high_equity_decisions': int((test.erp > threshold).sum()),
                   'zero_threshold_high_equity_decisions': int((test.erp > 0).sum()),
                   'gold_positive_decisions': int((test.gold_momentum_120_sessions > 0).sum()),
                   'erp_source_lag': 'Strictly earlier date; maximum seven calendar days old',
                   'status': 'Retrospective split; historical data vintage is not point-in-time certified',
                   'currency': 'CNY', 'copper': 'disabled: no validated CNY investable copper return series',
                   'cost': '0/5/10 bp per purchased or sold notional; terminal inventory marked to close',
                   'execution': 'next observed trading day open; monthly rebalance; fractional ETF units',
                   'annualization': '252 trading sessions; Sharpe risk-free rate zero'}
    (out_dir / 'assumptions.json').write_text(json.dumps(assumptions, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,5))
    for name in ['erp_gold','erp_only','6040_gold','6040','equity_buy_hold','bond_buy_hold']:
        ax.plot(navs[name].index, navs[name], label=name)
    ax.set(title='CNY ETF historical test, 5 bp per side (2023-2026)', ylabel='NAV, initial capital = 1')
    ax.legend(ncol=2); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(out_dir / 'test_nav.png', dpi=150); plt.close(fig)
    print(table.loc[table.cost_bps_per_side == 5].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'validated_20260930')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'validated_20260930')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
