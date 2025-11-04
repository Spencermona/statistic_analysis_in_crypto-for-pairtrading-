"""
加密货币配对交易统计分析脚本
功能：
1. 从Yahoo Finance获取前N名加密货币（排除稳定币）
2. 分别用BTC-USD和ETH-USD作为基准，与其他币种配对分析
3. 计算完整的统计模型指标（OLS、协整、ADF、半衰期等）
4. 生成配对分析结果CSV
5. 为最佳配对生成回测结果和信号
6. 输出图表：价格对比、残差z-score、回测权益曲线
"""

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import pearsonr, jarque_bera, skew, kurtosis
import matplotlib.pyplot as plt
from typing import List, Dict
import json
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ==================== 配置参数 ====================
TOP_N_CRYPTOS = 50              # 获取前N名加密货币
MIN_HISTORY_DAYS = 365 * 2      # 至少2年历史数据
START_DATE = "2020-01-01"       # 数据开始日期
END_DATE = datetime.now().strftime('%Y-%m-%d')  # 数据结束日期

# 排除的稳定币
EXCLUDE_STABLECOINS = {
    "USDT-USD", "USDC-USD", "BUSD-USD", "DAI-USD", "TUSD-USD",
    "FDUSD-USD", "PYUSD-USD", "EURT-USD"
}

# 交易策略参数
Z_ENTER = 2.0               # 入场阈值: |z| >= 2.0
Z_EXIT = 0.5                # 出场阈值: |z| <= 0.5
Z_STOP = 3.0                # 止损阈值: |z| >= 3.0
TIME_STOP_FACTOR = 1.0      # 时间止损 = 半衰期 * 系数

# 输出文件名
OUTPUT_BTC_CSV = "pair_scan_results_BTC_USD.csv"
OUTPUT_ETH_CSV = "pair_scan_results_ETH_USD.csv"
OUTPUT_COMBINED_CSV = "combined_pair_analysis.csv"
OUTPUT_BTC_SIGNAL_JSON = "trading_signals_BTC.json"
OUTPUT_ETH_SIGNAL_JSON = "trading_signals_ETH.json"


# ==================== 辅助函数 ====================

def get_top_crypto_tickers(top_n=50):
    """
    从Yahoo Finance获取前N名加密货币
    """
    YA_CRYPTO_URL = "https://finance.yahoo.com/cryptocurrencies/?count=250&offset=0"
    
    fallback_tickers = [
        "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "TRX-USD", "AVAX-USD",
        "LINK-USD", "MATIC-USD", "DOT-USD", "LTC-USD", "NEAR-USD", "BCH-USD", "ATOM-USD",
        "UNI7083-USD", "ETC-USD", "XLM-USD", "XMR-USD", "APT-USD", "OP-USD", "ALGO-USD",
        "FIL-USD", "HBAR-USD", "ICP-USD", "SUI-USD", "INJ-USD", "IMX-USD", "ARB-USD",
        "AAVE-USD", "EOS-USD", "MKR-USD", "FTM-USD", "EGLD-USD", "RUNE-USD", "THETA-USD",
        "VET-USD", "SAND-USD", "MANA-USD", "AXS-USD", "GRT-USD", "FLOW-USD", "XTZ-USD"
    ]
    
    try:
        tables = pd.read_html(YA_CRYPTO_URL)
        table = None
        for t in tables:
            if any(col for col in t.columns if str(col).lower() == 'symbol'):
                table = t
                break
        
        if table is None:
            raise ValueError("No Symbol table found on Yahoo page")
        
        # 找到Symbol列
        if 'Symbol' not in table.columns:
            sym_col = [c for c in table.columns if str(c).lower() == 'symbol'][0]
        else:
            sym_col = 'Symbol'
        
        syms = table[sym_col].astype(str).tolist()
        syms = [s for s in syms if s.endswith('-USD')]
        syms = list(dict.fromkeys(syms))  # 去重
        
        if not syms:
            raise ValueError("Empty symbols from Yahoo page")
        
        print(f"✓ 成功从Yahoo Finance获取 {len(syms[:top_n])} 个加密货币")
        return syms[:top_n]
    
    except Exception as e:
        print(f"⚠ Yahoo Finance抓取失败: {e}. 使用备用列表")
        return fallback_tickers[:top_n]


def calculate_halflife_ou(resid: pd.Series) -> float:
    """
    计算OU过程的半衰期
    """
    r = resid.dropna()
    if len(r) < 5:
        return np.nan
    
    lag = r.shift(1)
    delta = r - lag
    df = pd.concat([delta.rename("delta"), lag.rename("lag")], axis=1).dropna()
    
    if df.shape[0] < 5:
        return np.nan
    
    X = sm.add_constant(df["lag"])
    model = sm.OLS(df["delta"], X).fit()
    b = model.params.get("lag", np.nan)
    
    # 如果b >= 0, 没有均值回归
    if not np.isfinite(b) or b >= 0:
        return np.inf
    
    hl = -np.log(2.0) / b
    return float(hl) if np.isfinite(hl) and hl > 0 else np.inf


def compute_pair_metrics(base_symbol: str, prices_df: pd.DataFrame, min_days: int = 730) -> pd.DataFrame:
    """
    计算基准货币与所有其他货币的配对指标
    
    返回指标:
    - 基础统计: ticker, n_obs, corr, cov
    - OLS回归: alpha, beta, r2, adj_r2, f_stat, f_pvalue
    - 残差诊断: jb_stat, jb_pvalue, skew, kurt, dw_stat
    - 协整检验: coint_t, coint_p
    - 平稳性检验: adf_stat, adf_p
    - 半衰期: half_life
    - Z-score统计: z_mean, z_std, z_current, z_gt1, z_gt2, z_lt_1, z_lt_2
    - 综合评分: score, rating
    """
    cols = [c for c in prices_df.columns if c != base_symbol]
    results: List[Dict] = []
    
    base = prices_df[base_symbol].dropna()
    
    for ticker in cols:
        y = prices_df[ticker].dropna()
        pair = pd.concat([base.rename(base_symbol), y.rename(ticker)], axis=1).dropna()
        
        if pair.shape[0] < min_days:
            continue
        
        # === OLS回归: y ~ const + base ===
        X = sm.add_constant(pair[base_symbol])
        mdl = sm.OLS(pair[ticker], X).fit()
        alpha = float(mdl.params.get("const", np.nan))
        beta = float(mdl.params.get(base_symbol, np.nan))
        r2 = float(mdl.rsquared)
        adj_r2 = float(mdl.rsquared_adj)
        f_stat = float(getattr(mdl, "fvalue", np.nan))
        f_pvalue = float(getattr(mdl, "f_pvalue", np.nan))
        
        resid = mdl.resid
        
        # === 残差诊断 ===
        jb_stat, jb_p = jarque_bera(resid)
        dw = durbin_watson(resid)
        sk = float(skew(resid, bias=False))
        ku = float(kurtosis(resid, fisher=True, bias=False))
        
        # === 协整检验 (Engle-Granger) ===
        try:
            c_t, c_p, _crit = coint(pair[base_symbol], pair[ticker])
        except Exception:
            c_t, c_p = np.nan, np.nan
        
        # === ADF检验（残差平稳性） ===
        try:
            adf_res = adfuller(resid.dropna(), autolag="AIC")
            adf_stat = float(adf_res[0])
            adf_p = float(adf_res[1])
        except Exception:
            adf_stat, adf_p = np.nan, np.nan
        
        # === 相关性和协方差 ===
        try:
            corr, _ = pearsonr(pair[base_symbol], pair[ticker])
        except Exception:
            corr = float(pair[base_symbol].corr(pair[ticker]))
        cov = float(pair[base_symbol].cov(pair[ticker]))
        
        # === 半衰期 ===
        hl = calculate_halflife_ou(resid)
        
        # === Z-score序列 ===
        z = (resid - resid.mean()) / resid.std(ddof=0)
        z_mean = float(z.mean())
        z_std = float(z.std(ddof=0))
        z_current = float(z.iloc[-1])
        z_gt1 = float((z > 1).mean())
        z_gt2 = float((z > 2).mean())
        z_lt_1 = float((z < -1).mean())
        z_lt_2 = float((z < -2).mean())
        
        # === 综合评分 (0-100) ===
        corr_score = np.clip((corr - 0.5) / 0.5, 0, 1) * 25       # 相关性 >= 0.5
        coint_score = np.clip((0.3 - (c_p if np.isfinite(c_p) else 1)) / 0.3, 0, 1) * 30  # 协整p <= 0.3
        adf_score = np.clip((0.1 - (adf_p if np.isfinite(adf_p) else 1)) / 0.1, 0, 1) * 25  # ADF p <= 0.1
        hl_score = np.clip((120 - (hl if np.isfinite(hl) else 1e6)) / 120, 0, 1) * 20  # 半衰期 <= 120天
        score = int(round(corr_score + coint_score + adf_score + hl_score))
        
        # === 评级 ===
        rating = (
            "Good" if (corr >= 0.8 and (c_p if np.isfinite(c_p) else 1) < 0.05 and 
                      (adf_p if np.isfinite(adf_p) else 1) < 0.05 and hl < 60)
            else "Fair" if (corr >= 0.7 and (c_p if np.isfinite(c_p) else 1) < 0.1 and 
                           (adf_p if np.isfinite(adf_p) else 1) < 0.1 and hl < 120)
            else "Weak"
        )
        
        results.append({
            "ticker": ticker,
            "n_obs": int(pair.shape[0]),
            "corr": float(corr),
            "cov": float(cov),
            "alpha": alpha,
            "beta": beta,
            "r2": r2,
            "adj_r2": adj_r2,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "jb_stat": float(jb_stat),
            "jb_pvalue": float(jb_p),
            "skew": sk,
            "kurt": ku,
            "dw_stat": float(dw),
            "coint_t": float(c_t) if np.isfinite(c_t) else np.nan,
            "coint_p": float(c_p) if np.isfinite(c_p) else np.nan,
            "adf_stat": adf_stat,
            "adf_p": adf_p,
            "half_life": float(hl) if np.isfinite(hl) else np.inf,
            "z_mean": z_mean,
            "z_std": z_std,
            "z_current": z_current,
            "z_gt1": z_gt1,
            "z_gt2": z_gt2,
            "z_lt_1": z_lt_1,
            "z_lt_2": z_lt_2,
            "score": score,
            "rating": rating,
        })
    
    df_out = pd.DataFrame(results).sort_values(["score", "corr"], ascending=[False, False])
    return df_out.reset_index(drop=True)


def backtest_pair(prices: pd.DataFrame, base: str, cand: str,
                  z_enter=2.0, z_exit=0.5, z_stop=3.0,
                  time_stop_days=60):
    """
    配对交易回测
    """
    px_base = prices[base].dropna()
    px_cand = prices[cand].dropna()
    df = pd.concat([px_base.rename(base), px_cand.rename(cand)], axis=1).dropna()
    
    X = sm.add_constant(df[base])
    mdl = sm.OLS(df[cand], X).fit()
    beta = float(mdl.params[base])
    resid = mdl.resid
    z = (resid - resid.mean()) / resid.std(ddof=0)
    
    # 日收益率
    ret = df.pct_change().dropna()
    dates = ret.index
    
    pos_c = 0.0  # 候选币仓位
    pos_b = 0.0  # 基准币仓位
    entry_date = None
    pnl = []
    equity = [1.0]
    max_equity = 1.0
    dd = [0.0]
    trades = []
    
    def target_weights(zv):
        """根据z-score计算目标仓位"""
        if zv >= z_enter:
            # 做空候选币，做多基准币
            wc = -1.0
            wb = +beta
        elif zv <= -z_enter:
            # 做多候选币，做空基准币
            wc = +1.0
            wb = -beta
        else:
            return 0.0, 0.0
        scale = 1.0 / (abs(wc) + abs(wb)) if (abs(wc) + abs(wb)) > 0 else 0.0
        return wc * scale, wb * scale
    
    holding = 0
    for i, dt in enumerate(dates):
        if dt not in z.index:
            pnl.append(0.0)
            equity.append(equity[-1])
            dd.append((max_equity - equity[-1]) / max_equity)
            continue
        
        zt = float(z.loc[dt])
        
        # 入场/出场/止损逻辑
        if pos_c == 0.0 and pos_b == 0.0:
            wc, wb = target_weights(zt)
            if wc != 0.0:
                pos_c, pos_b = wc, wb
                entry_date = dt
                holding = 0
        else:
            holding += 1
            # 硬止损
            if abs(zt) >= z_stop:
                trades.append({
                    "entry": entry_date,
                    "exit": dt,
                    "reason": "hard_stop",
                    "holding": holding
                })
                pos_c = pos_b = 0.0
                holding = 0
            # 时间止损
            elif holding >= time_stop_days:
                trades.append({
                    "entry": entry_date,
                    "exit": dt,
                    "reason": "time_stop",
                    "holding": holding
                })
                pos_c = pos_b = 0.0
                holding = 0
            # 均值回归出场
            elif abs(zt) <= z_exit:
                trades.append({
                    "entry": entry_date,
                    "exit": dt,
                    "reason": "revert",
                    "holding": holding
                })
                pos_c = pos_b = 0.0
                holding = 0
        
        # PnL计算
        rc = ret.iloc[i][cand] if i < len(ret) else 0.0
        rb = ret.iloc[i][base] if i < len(ret) else 0.0
        day_pnl = pos_c * rc + pos_b * rb
        pnl.append(day_pnl)
        eq = equity[-1] * (1.0 + day_pnl)
        equity.append(eq)
        max_equity = max(max_equity, eq)
        dd.append((max_equity - eq) / max_equity)
    
    equity = pd.Series(equity[1:], index=dates)
    pnl = pd.Series(pnl, index=dates)
    dd = pd.Series(dd[1:], index=dates)
    
    kpis = {}
    kpis['total_return'] = equity.iloc[-1] - 1.0
    kpis['ann_return'] = (equity.iloc[-1]) ** (252 / len(equity)) - 1.0 if len(equity) > 0 else 0.0
    kpis['ann_vol'] = pnl.std(ddof=0) * np.sqrt(252)
    kpis['sharpe'] = (pnl.mean() / pnl.std(ddof=0)) * np.sqrt(252) if pnl.std(ddof=0) > 0 else 0.0
    kpis['max_dd'] = dd.max()
    kpis['trades'] = len(trades)
    kpis['avg_holding_days'] = np.mean([t['holding'] for t in trades]) if trades else 0
    
    return {
        'beta': beta,
        'equity': equity,
        'pnl': pnl,
        'drawdown': dd,
        'trades': pd.DataFrame(trades),
        'kpis': kpis,
        'z_series': z
    }


def generate_trading_signal(base: str, cand: str, beta: float, z_now: float, 
                           hl_days: float, z_enter: float, z_exit: float, 
                           z_stop: float, time_stop_factor: float):
    """
    生成交易信号
    """
    # 方向判断
    if z_now >= z_enter:
        direction = f"Short {cand} / Long {base}"
        position_cand = "SHORT"
        position_base = "LONG"
    elif z_now <= -z_enter:
        direction = f"Long {cand} / Short {base}"
        position_cand = "LONG"
        position_base = "SHORT"
    else:
        direction = "No trade (|z| below entry)"
        position_cand = "FLAT"
        position_base = "FLAT"
    
    # 时间止损天数
    time_stop_days = math.ceil((hl_days if np.isfinite(hl_days) else 60.0) * time_stop_factor)
    
    # 仓位权重计算（市场中性）
    if position_cand != "FLAT":
        total_weight = abs(1.0) + abs(beta)
        weight_cand = 1.0 / total_weight if position_cand == "LONG" else -1.0 / total_weight
        weight_base = -beta / total_weight if position_cand == "LONG" else beta / total_weight
    else:
        weight_cand = 0.0
        weight_base = 0.0
    
    signal = {
        "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
        "base": base,
        "candidate": cand,
        "beta": round(beta, 4),
        "z_current": round(z_now, 4),
        "half_life_days": round(hl_days, 2) if np.isfinite(hl_days) else None,
        "direction": direction,
        "position_candidate": position_cand,
        "position_base": position_base,
        "weight_candidate": round(weight_cand, 4),
        "weight_base": round(weight_base, 4),
        "entry_threshold": z_enter,
        "exit_threshold": z_exit,
        "stop_loss_threshold": z_stop,
        "time_stop_days": time_stop_days,
        "time_stop_months": round(time_stop_days / 21, 2)
    }
    
    return signal


def plot_pair_analysis(base: str, cand: str, prices_df: pd.DataFrame, 
                       z_series: pd.Series, beta: float, corr: float,
                       save_path: str):
    """
    绘制配对分析图：价格对比 + z-score
    """
    x = prices_df[base].dropna()
    y = prices_df[cand].dropna()
    df_pair = pd.concat([x.rename(base), y.rename(cand)], axis=1).dropna()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # 标准化价格对比
    ax1.plot(df_pair.index, df_pair[base] / df_pair[base].iloc[0], 
             label=base, linewidth=2, color='blue')
    ax1.plot(df_pair.index, df_pair[cand] / df_pair[cand].iloc[0], 
             label=cand, linewidth=2, color='orange')
    ax1.set_title(f"Price Comparison: {base} vs {cand} (beta={beta:.3f}, corr={corr:.3f})", 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_ylabel('Normalized Price', fontsize=11)
    
    # Z-score
    ax2.plot(z_series.index, z_series, color='purple', linewidth=1.5, label='Z-score')
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='-')
    ax2.axhline(Z_ENTER, color='red', linewidth=1.5, linestyle='--', label=f'Entry (+{Z_ENTER})')
    ax2.axhline(-Z_ENTER, color='red', linewidth=1.5, linestyle='--', label=f'Entry (-{Z_ENTER})')
    ax2.axhline(Z_EXIT, color='green', linewidth=1, linestyle='--', alpha=0.7, label=f'Exit (±{Z_EXIT})')
    ax2.axhline(-Z_EXIT, color='green', linewidth=1, linestyle='--', alpha=0.7)
    ax2.axhline(Z_STOP, color='darkred', linewidth=1, linestyle=':', alpha=0.7, label=f'Stop (±{Z_STOP})')
    ax2.axhline(-Z_STOP, color='darkred', linewidth=1, linestyle=':', alpha=0.7)
    ax2.fill_between(z_series.index, -Z_ENTER, Z_ENTER, alpha=0.1, color='green')
    ax2.set_title("Residual Z-score", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylabel('Z-score', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 配对分析图已保存: {save_path}")
    plt.close()


def plot_backtest_results(base: str, cand: str, backtest_res: dict, save_path: str):
    """
    绘制回测结果：权益曲线 + 回撤
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # 权益曲线
    equity = backtest_res['equity']
    ax1.plot(equity.index, equity.values, linewidth=2, color='green')
    ax1.set_title(f"Backtest Equity Curve: {base} vs {cand}", fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_ylabel('Equity', fontsize=11)
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    
    # 回撤
    dd = backtest_res['drawdown']
    ax2.plot(dd.index, dd.values, color='crimson', linewidth=2)
    ax2.fill_between(dd.index, 0, dd.values, color='crimson', alpha=0.3)
    ax2.set_title("Drawdown", fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylabel('Drawdown', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    
    # 添加KPI文本
    kpis = backtest_res['kpis']
    kpi_text = (
        f"Total Return: {kpis['total_return']:.2%}\n"
        f"Ann. Return: {kpis['ann_return']:.2%}\n"
        f"Ann. Vol: {kpis['ann_vol']:.2%}\n"
        f"Sharpe Ratio: {kpis['sharpe']:.2f}\n"
        f"Max Drawdown: {kpis['max_dd']:.2%}\n"
        f"Trades: {kpis['trades']}\n"
        f"Avg Holding: {kpis['avg_holding_days']:.1f} days"
    )
    ax1.text(0.02, 0.98, kpi_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 回测结果图已保存: {save_path}")
    plt.close()


# ==================== 主流程 ====================

def main():
    print("="*80)
    print("加密货币配对交易统计分析")
    print("="*80)
    
    # Step 1: 获取加密货币列表
    print("\n[Step 1] 获取Yahoo Finance前{}名加密货币...".format(TOP_N_CRYPTOS))
    all_tickers = get_top_crypto_tickers(TOP_N_CRYPTOS)
    
    # 排除BTC、ETH和稳定币
    candidates = [t for t in all_tickers 
                  if t not in ["BTC-USD", "ETH-USD"] and t not in EXCLUDE_STABLECOINS]
    print(f"✓ 候选币种数量: {len(candidates)}")
    print(f"  前10个: {candidates[:10]}")
    
    # Step 2: 下载价格数据
    print(f"\n[Step 2] 下载价格数据 (从 {START_DATE} 到 {END_DATE})...")
    base_list = ["BTC-USD", "ETH-USD"]
    union_tickers = sorted(set(candidates) | set(base_list))
    print(f"  总计需下载: {len(union_tickers)} 个币种")
    
    try:
        raw = yf.download(union_tickers, start=START_DATE, end=END_DATE, 
                         interval="1d", group_by='ticker', progress=True, 
                         auto_adjust=False, threads=True)
        
        prices_union = pd.DataFrame()
        for t in union_tickers:
            try:
                if len(union_tickers) == 1:
                    s = raw['Close'].rename(t).dropna()
                else:
                    s = raw[t]['Close'].rename(t).dropna()
                prices_union = pd.concat([prices_union, s], axis=1)
            except Exception:
                pass
    except Exception as e:
        print(f"⚠ 批量下载失败: {e}. 使用单个下载...")
        cols = {}
        for t in union_tickers:
            try:
                df = yf.download(t, start=START_DATE, end=END_DATE, progress=False)
                cols[t] = df['Close'].rename(t)
            except Exception as ee:
                print(f"  跳过 {t}: {ee}")
        prices_union = pd.DataFrame(cols)
    
    # 过滤数据不足的币种
    valid_cols = [c for c in prices_union.columns 
                  if prices_union[c].dropna().shape[0] >= MIN_HISTORY_DAYS]
    prices_union = prices_union[valid_cols]
    print(f"✓ 价格数据框维度: {prices_union.shape} (有效币种: {len(valid_cols)})")
    
    # Step 3: 计算配对指标
    print("\n[Step 3] 计算BTC-USD和ETH-USD的配对指标...")
    combined = []
    
    for base in base_list:
        if base not in prices_union.columns:
            print(f"⚠ {base} 不在价格数据中，跳过")
            continue
        
        print(f"\n  分析 {base}...")
        df_metrics = compute_pair_metrics(base, prices_union, MIN_HISTORY_DAYS)
        
        # 排除BTC和ETH互相配对
        if base == "BTC-USD":
            df_metrics = df_metrics[df_metrics["ticker"] != "ETH-USD"]
            output_file = OUTPUT_BTC_CSV
        elif base == "ETH-USD":
            df_metrics = df_metrics[df_metrics["ticker"] != "BTC-USD"]
            output_file = OUTPUT_ETH_CSV
        
        df_metrics.to_csv(output_file, index=False)
        print(f"  ✓ 保存: {output_file} ({len(df_metrics)} 个配对)")
        
        df_metrics['base'] = base
        combined.append(df_metrics)
        
        # 显示前5个最佳配对
        if not df_metrics.empty:
            print(f"\n  {base} 前5个最佳配对:")
            display_cols = ['ticker', 'corr', 'coint_p', 'adf_p', 'half_life', 'beta', 'score', 'z_current', 'rating']
            print(df_metrics[display_cols].head(5).to_string(index=False))
    
    # 合并结果
    if not combined:
        print("\n❌ 没有可用的配对分析结果")
        return
    
    combo = pd.concat(combined, ignore_index=True)
    combo_sorted = combo.sort_values(['score', 'corr'], ascending=[False, False])
    combo_sorted.to_csv(OUTPUT_COMBINED_CSV, index=False)
    print(f"\n✓ 合并结果已保存: {OUTPUT_COMBINED_CSV} ({len(combo_sorted)} 个配对)")
    
    # Step 4: 为BTC和ETH各自生成最佳配对的信号和回测
    print("\n[Step 4] 为BTC和ETH各自的最佳配对生成交易信号和回测...")
    
    for base in base_list:
        if base not in prices_union.columns:
            continue
            
        # 获取该基准币的最佳配对
        base_best = combo_sorted[combo_sorted['base'] == base].iloc[0]
        CAND = str(base_best['ticker'])
        score = int(base_best.get('score', -1))
        corr = float(base_best.get('corr', float('nan')))
        beta = float(base_best.get('beta', float('nan')))
        hl_days = float(base_best.get('half_life', 60.0))
        
        print(f"\n{'='*80}")
        print(f"处理 {base} 的最佳配对")
        print(f"{'='*80}")
        print(f"  配对: {base} vs {CAND}")
        print(f"  评分: {score} | 相关性: {corr:.3f} | Beta: {beta:.3f} | 半衰期: {hl_days:.1f}天")
        
        # 生成交易信号
        print(f"\n  [4.1] 生成 {base} 交易信号...")
        x = prices_union[base].dropna()
        y = prices_union[CAND].dropna()
        df_pair = pd.concat([x.rename(base), y.rename(CAND)], axis=1).dropna()
        X = sm.add_constant(df_pair[base])
        mdl = sm.OLS(df_pair[CAND], X).fit()
        resid = mdl.resid
        z = (resid - resid.mean()) / resid.std(ddof=0)
        z_now = float(z.iloc[-1])
        
        signal = generate_trading_signal(base, CAND, beta, z_now, hl_days, 
                                         Z_ENTER, Z_EXIT, Z_STOP, TIME_STOP_FACTOR)
        
        # 保存信号
        signal_file = OUTPUT_BTC_SIGNAL_JSON if base == "BTC-USD" else OUTPUT_ETH_SIGNAL_JSON
        with open(signal_file, 'w', encoding='utf-8') as f:
            json.dump(signal, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 交易信号已保存: {signal_file}")
        
        # 打印信号
        print(f"\n  交易信号详情:")
        print(f"  {'─'*76}")
        for k, v in signal.items():
            print(f"    {k}: {v}")
        print(f"  {'─'*76}")
        
        # 执行回测
        print(f"\n  [4.2] 执行 {base} 回测...")
        time_stop_days = int(np.ceil(hl_days * TIME_STOP_FACTOR))
        backtest_res = backtest_pair(prices_union, base, CAND,
                                      z_enter=Z_ENTER, z_exit=Z_EXIT, 
                                      z_stop=Z_STOP, time_stop_days=time_stop_days)
        
        print(f"\n  回测KPI:")
        for k, v in backtest_res['kpis'].items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
        
        # 生成图表
        print(f"\n  [4.3] 生成 {base} 图表...")
        
        # 配对分析图
        plot_pair_analysis(base, CAND, prices_union, backtest_res['z_series'], 
                          beta, corr, f"pair_analysis_{base}_{CAND}.png")
        
        # 回测结果图
        plot_backtest_results(base, CAND, backtest_res, 
                             f"backtest_results_{base}_{CAND}.png")
    
    print("\n" + "="*80)
    print("✓ 分析完成！")
    print("="*80)
    print("\n生成的文件:")
    print(f"  1. {OUTPUT_BTC_CSV} - BTC配对分析结果")
    print(f"  2. {OUTPUT_ETH_CSV} - ETH配对分析结果")
    print(f"  3. {OUTPUT_COMBINED_CSV} - 合并配对分析结果")
    print(f"  4. {OUTPUT_BTC_SIGNAL_JSON} - BTC交易信号")
    print(f"  5. {OUTPUT_ETH_SIGNAL_JSON} - ETH交易信号")
    print(f"  6. pair_analysis_BTC-USD_*.png - BTC配对分析图表")
    print(f"  7. pair_analysis_ETH-USD_*.png - ETH配对分析图表")
    print(f"  8. backtest_results_BTC-USD_*.png - BTC回测结果图表")
    print(f"  9. backtest_results_ETH-USD_*.png - ETH回测结果图表")
    print("="*80)


if __name__ == "__main__":
    main()
