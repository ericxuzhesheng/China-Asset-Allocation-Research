import numpy as np
import pandas as pd
import pytest
from allocation import ASSETS, backtest, load_prices, metrics, signals


def prices(open_values, close_values, dates=None):
    if dates is None:
        dates = pd.bdate_range('2024-01-02', periods=len(open_values))
    return pd.concat({'open': pd.DataFrame(open_values, index=dates, columns=ASSETS),
                      'close': pd.DataFrame(close_values, index=dates, columns=ASSETS)}, axis=1)


def test_same_period_benchmark_and_drift():
    p = prices([[100,100,100],[110,100,100]], [[110,100,100],[121,100,100]])
    schedule = pd.DataFrame([[.6,.4,0]], columns=ASSETS, index=p.index[:1])
    result = backtest(p, schedule, 0)
    assert result.nav.iloc[0] == pytest.approx(1.06)
    assert result.nav.iloc[1] == pytest.approx(1.126)
    assert result['return'].iloc[1] == pytest.approx(1.126/1.06-1)
    assert result['weight_510300.SH'].iloc[1] == pytest.approx(.66/1.06)


def test_self_financing_costs_buy_and_sell():
    p = prices([[100]*3]*2, [[100]*3]*2)
    schedule = pd.DataFrame([[1,0,0],[0,1,0]], columns=ASSETS, index=p.index)
    result = backtest(p, schedule, 10)
    assert result.nav.iloc[0] == pytest.approx(1/1.001)
    assert result.nav.iloc[1] == pytest.approx((1/1.001)*.999/1.001)
    assert result.cost_cash.sum() == pytest.approx(1-result.nav.iloc[-1])


def test_daily_momentum_strictly_prior_erp_next_open():
    dates = pd.bdate_range('2023-01-02', periods=160)
    values = np.tile(np.arange(100,260)[:,None], (1,3))
    p = prices(values, values, dates)
    erp = pd.DataFrame({'erp': np.arange(len(dates))/10000}, index=dates)
    s = signals(p, erp)
    first = s.iloc[0]
    i = dates.get_loc(first.decision_date)
    assert first.gold_momentum_120_sessions == pytest.approx((100+i)/(100+i-120)-1)
    assert first.erp_source_date < first.decision_date < s.index[0]
    assert first.erp == pytest.approx(erp.loc[dates[i-1], 'erp'])
    altered = p.copy()
    altered.loc[s.index[0]:, ('close',ASSETS[2])] *= 9
    assert signals(altered, erp).iloc[0].equals(first)


def test_drawdown_includes_initial_capital():
    assert metrics(pd.Series([-.2,.1]))['max_drawdown'] == pytest.approx(-.2)


def test_missing_erp_fails_instead_of_indefinite_fill():
    dates = pd.bdate_range('2023-01-02', periods=160)
    p = prices([[100]*3]*160, [[100]*3]*160, dates)
    with pytest.raises(ValueError, match='Missing fresh ERP'):
        signals(p, pd.DataFrame({'erp':[.03]}, index=dates[:1]))


def test_missing_or_impossible_asset_prices_are_rejected(tmp_path):
    path = tmp_path / 'prices.csv'
    rows = [{'date':'2024-01-02', 'symbol':s, 'open':100, 'close':100, 'high':101, 'low':99} for s in ASSETS]
    pd.DataFrame(rows[:-1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='cover all three'):
        load_prices(path)
    rows[-1]['high'] = 90
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Impossible OHLC'):
        load_prices(path)
