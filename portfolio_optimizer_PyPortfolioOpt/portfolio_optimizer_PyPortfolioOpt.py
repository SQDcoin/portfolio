import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager, rc
from matplotlib.backends.backend_pdf import PdfPages
import platform

# PyPortfolioOpt 라이브러리 임포트
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting

#################################################################
## CONFIGURATION (설정)
#################################################################
# (설정 영역은 이전과 동일)
INITIAL_TICKERS = [
    'SPY', 'EFA', 'AGG', 'TLT', 'TIP', 'GLD', 'IJR', 'EEM',
    'VNQ', 'DBC', 'DJP', 'USO', 'SLV', 'CPER', 'VXX',
    'CUT', 'PSP', 'BTC-USD', 'ETH-USD',
    # 'QAI'
]
MIN_YEARS_OF_DATA = 15
DATA_INTERVAL = '1mo'
RISK_FREE_RATE = 0.025
OPTIMIZER_SOLVER = 'ECOS'
CONSERVATIVE_MIX_RATIO = 0.7
ROLLING_WINDOW_YEARS = 10
ROLLING_STEP_YEARS = 1
# ★★★ 변경점: 출력 파일명은 이제 폴더 내에 생성되므로 기본 이름만 설정 ★★★
OUTPUT_PDF_FILENAME = 'cml_report.pdf'
OUTPUT_CHART_FILENAME = 'rolling_composition.png'
OUTPUT_EXCEL_FILENAME = 'analysis_report.xlsx'
#################################################################

# --- 한글 폰트 설정 ---
if platform.system() == 'Windows':
    font_path = r'C:\Windows\Fonts\malgun.ttf'
elif platform.system() == 'Darwin':
    font_path = '/System/Library/Fonts/AppleGothic.ttf'
else:
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
mpl.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# --- 파이 차트 생성 함수 ---
def pie_chart(ax, weights, title):
    # '현금 (무위험 자산)' -> '현금' 으로 일관성 유지
    cash_label = '현금'
    cash = weights.pop(cash_label, 0)
    dfw = pd.Series(weights).to_frame('Weight')
    small = dfw[dfw['Weight'] < 0.01]
    other = small['Weight'].sum() if not small.empty else 0
    dfw.drop(small.index, inplace=True)
    final = dfw['Weight'].to_dict()
    if other > 0: final['기타'] = other
    if cash > 0: final[cash_label] = cash
    ax.pie(final.values(), labels=final.keys(), autopct='%1.1f%%', startangle=90)
    ax.set_title(title, fontfamily=font_name, fontsize=14)

# ★★★ 변경점: output_dir을 인자로 받고, summary_df를 반환하도록 수정 ★★★
def run_cml_optimization(prices, output_dir):
    print("\n" + "="*50)
    print("📈 단일 기간 전체 포트폴리오 분석 시작...")

    mu = expected_returns.mean_historical_return(prices, frequency=12)
    S = risk_models.sample_cov(prices, frequency=12)

    ef_max = EfficientFrontier(mu, S, solver=OPTIMIZER_SOLVER)
    w_max = ef_max.max_sharpe(risk_free_rate=RISK_FREE_RATE)
    perf_max = ef_max.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

    ef_min = EfficientFrontier(mu, S, solver=OPTIMIZER_SOLVER)
    w_min = ef_min.min_volatility()
    perf_min = ef_min.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

    final_ret = perf_max[0] * CONSERVATIVE_MIX_RATIO + RISK_FREE_RATE * (1 - CONSERVATIVE_MIX_RATIO)
    final_vol = perf_max[1] * CONSERVATIVE_MIX_RATIO
    final_sharpe = (final_ret - RISK_FREE_RATE) / final_vol if final_vol > 0 else 0
    perf_cons = (final_ret, final_vol, final_sharpe)
    w_cons = {k: v * CONSERVATIVE_MIX_RATIO for k, v in w_max.items()}
    w_cons['현금'] = 1 - CONSERVATIVE_MIX_RATIO

    portfolios = {
        "🎯 최대 샤프 (원액)": (w_max, perf_max), "🛡️ 최소 변동성": (w_min, perf_min), "👵 어머님 추천": (w_cons, perf_cons)
    }
    print("\n👵 어머님을 위한 최종 포트폴리오 분석 결과")
    for name, (weights, perf) in portfolios.items():
        print(f"\n--- {name} ---")
        print(f"기대 수익률: {perf[0]:.2%}, 예상 변동성: {perf[1]:.2%}, 샤프 지수: {perf[2]:.2f}")
        print("자산 비중:")
        for tk, wt in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            if wt > 0.0001: print(f"  - {tk}: {wt:.1%}")

    pdf_path = os.path.join(output_dir, OUTPUT_PDF_FILENAME)
    print(f"\n📄 PDF 리포트 생성 중... ({pdf_path})")
    with PdfPages(pdf_path) as pdf:
        # ... (PDF 생성 로직은 동일) ...
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        plotting.plot_efficient_frontier(EfficientFrontier(mu, S, solver=OPTIMIZER_SOLVER), ax=ax1, show_assets=False)
        x = np.linspace(0, perf_max[1] * 1.5, 100)
        y = RISK_FREE_RATE + x * (perf_max[0] - RISK_FREE_RATE) / perf_max[1]
        ax1.plot(x, y, 'k--', label='자본배분선 (CAL)')
        ax1.scatter(perf_max[1], perf_max[0], marker='*', s=200, c='r', label='최대 샤프')
        ax1.scatter(perf_min[1], perf_min[0], marker='P', s=150, c='g', label='최소 변동성')
        ax1.scatter(final_vol, final_ret, marker='*', s=200, c='b', label='어머님 추천')
        ax1.legend(prop={'family': font_name})
        ax1.set_title('효율적 투자선과 포트폴리오', fontfamily=font_name)
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)

        fig2, axes = plt.subplots(1, 3, figsize=(24, 8))
        pie_chart(axes[0], w_max.copy(), "최대 샤프")
        pie_chart(axes[1], w_min.copy(), "최소 변동성")
        pie_chart(axes[2], w_cons.copy(), "어머님 추천")
        plt.suptitle('포트폴리오 자산 구성 비교', fontfamily=font_name, fontsize=18)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
    print("PDF 리포트 생성 완료.")

    df_summary = pd.DataFrame([
        {'Portfolio': n, 'Return': p[0], 'Volatility': p[1], 'Sharpe': p[2]}
        for n, (_, p) in portfolios.items()
    ]).set_index('Portfolio')
    return df_summary

def run_rolling_analysis_all_assets(prices, window_years, step_years):
    print("\n" + "="*50)
    print(f"⏳ {window_years}년 단위 롤링 분석 시작 (매 {step_years}년 이동)...")

    all_years = sorted(prices.index.year.unique())
    start_years = range(all_years[0], all_years[-1] - window_years + 2, step_years)
    
    weights_list = []
    performance_list = []

    for start_year in start_years:
        start_date = f"{start_year}-01-01"
        end_date = f"{start_year + window_years -1}-12-31"
        period_prices = prices.loc[start_date:end_date]
        if len(period_prices) < 24: continue
        print(f"  - 분석 기간: {period_prices.index[0].date()} ~ {period_prices.index[-1].date()}")

        try:
            mu = expected_returns.mean_historical_return(period_prices, frequency=12)
            S = risk_models.sample_cov(period_prices, frequency=12)
            ef = EfficientFrontier(mu, S, solver=OPTIMIZER_SOLVER)
            
            w_max = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
            perf_max = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
            
            w_cons = {ticker: weight * CONSERVATIVE_MIX_RATIO for ticker, weight in w_max.items()}
            w_cons['현금'] = 1 - CONSERVATIVE_MIX_RATIO
            
            cleaned_weights = {ticker: weight for ticker, weight in w_cons.items() if weight > 0.001}
            cleaned_weights['end_year'] = start_year + window_years - 1
            weights_list.append(cleaned_weights)
            
            cons_return = perf_max[0] * CONSERVATIVE_MIX_RATIO + RISK_FREE_RATE * (1 - CONSERVATIVE_MIX_RATIO)
            cons_volatility = perf_max[1] * CONSERVATIVE_MIX_RATIO
            cons_sharpe = (cons_return - RISK_FREE_RATE) / cons_volatility if cons_volatility > 0 else 0
            
            performance_list.append({
                'end_year': start_year + window_years - 1,
                'Return': cons_return,
                'Volatility': cons_volatility,
                'Sharpe': cons_sharpe
            })
        except Exception as e:
            print(f"    - 오류 발생: {e}")

    if not weights_list: return pd.DataFrame(), pd.DataFrame()
    
    weights_df = pd.DataFrame(weights_list).set_index('end_year').fillna(0)
    performance_df = pd.DataFrame(performance_list).set_index('end_year')
    
    return weights_df, performance_df

if __name__ == '__main__':
    # ★★★ 변경점: 모든 결과를 저장할 통합 폴더 생성 ★★★
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'portfolio_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ 모든 결과는 '{output_dir}' 폴더에 저장됩니다.")

    print(f"\n🔬 {MIN_YEARS_OF_DATA}년 이상 데이터를 가진 자산을 필터링합니다...")
    
    final_tickers = []
    excluded_tickers = []
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=MIN_YEARS_OF_DATA * 365)
    
    for ticker in INITIAL_TICKERS:
        try:
            df = yf.download(ticker, start='1990-01-01', end=datetime.datetime.now(), progress=False, timeout=5)
            if not df.empty and df.index[0] < cutoff_date:
                final_tickers.append(ticker)
            else:
                excluded_tickers.append(ticker)
        except Exception:
            excluded_tickers.append(ticker)
            
    print("\n--- 필터링 결과 ---")
    print(f"✅ 최종 분석 대상 ({len(final_tickers)}개): {', '.join(final_tickers)}")
    print(f"❌ 제외 대상 ({len(excluded_tickers)}개): {', '.join(excluded_tickers)}")

    print("\n최종 데이터를 다운로드합니다...")
    all_prices = pd.DataFrame()
    for t in final_tickers:
        df = yf.download(t, start='1990-01-01', end=datetime.datetime.now(), interval=DATA_INTERVAL, auto_adjust=True, progress=False)
        if not df.empty: all_prices[t] = df['Close']
    
    all_prices.ffill(inplace=True)
    first_valid_date = all_prices.dropna().index[0]
    all_prices = all_prices.loc[first_valid_date:]

    print("\n실제 분석 데이터 시작일:", all_prices.index[0].date())
    print("실제 분석 데이터 종료일:", all_prices.index[-1].date())

    if len(all_prices) < 24:
        print("오류: 분석에 필요한 데이터가 부족합니다.")
    else:
        summary_df = run_cml_optimization(all_prices, output_dir)

        rolling_weights_df, rolling_performance_df = run_rolling_analysis_all_assets(
            all_prices,
            window_years=ROLLING_WINDOW_YEARS,
            step_years=ROLLING_STEP_YEARS
        )

        if not rolling_weights_df.empty:
            print("\n[롤링 분석 최종 결과 (포트폴리오 성과)]")
            print(rolling_performance_df.to_string(
                formatters={'Return': '{:.2%}'.format, 'Volatility': '{:.2%}'.format, 'Sharpe': '{:.2f}'.format}
            ))
            
            print("\n[롤링 분석 최종 결과 (자산별 비중)]")
            if '현금' in rolling_weights_df.columns:
                cols = ['현금'] + [col for col in rolling_weights_df.columns if col != '현금']
                rolling_weights_df = rolling_weights_df[cols]
            print((rolling_weights_df * 100).round(1).to_string(float_format='%.1f%%'))
            
            chart_path = os.path.join(output_dir, OUTPUT_CHART_FILENAME)
            print(f"\n📊 롤링 분석 그래프 생성 중... ({chart_path})")
            fig, ax = plt.subplots(figsize=(15, 9))
            
            if '현금' in rolling_weights_df.columns:
                plot_df_risk = rolling_weights_df.drop(columns=['현금'])
                plot_df_risk = plot_df_risk.reindex(sorted(plot_df_risk.columns), axis=1)
                plot_df = pd.concat([rolling_weights_df[['현금']], plot_df_risk], axis=1)
            else:
                plot_df = rolling_weights_df.reindex(sorted(rolling_weights_df.columns), axis=1)

            ax.stackplot(plot_df.index, plot_df.T, labels=plot_df.columns, alpha=0.8)
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
            ax.set_title(f'기간별 최적 포트폴리오 자산 구성 변화 ({ROLLING_WINDOW_YEARS}년 롤링)', fontsize=18)
            ax.set_xlabel('분석 기간의 종료 연도', fontsize=12)
            ax.set_ylabel('자산 비중', fontsize=12)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="자산 Tickers")
            ax.grid(True, linestyle='--')
            
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(chart_path)
            plt.show()
            print("롤링 분석 그래프 생성 완료.")

        # ★★★ 변경점: 모든 수치 결과를 하나의 엑셀 파일로 저장 ★★★
        excel_path = os.path.join(output_dir, OUTPUT_EXCEL_FILENAME)
        print(f"\n💾 모든 수치 데이터를 엑셀 파일로 저장 중... ({excel_path})")
        try:
            with pd.ExcelWriter(excel_path) as writer:
                summary_df.to_excel(writer, sheet_name='CML_Summary')
                rolling_performance_df.to_excel(writer, sheet_name='Rolling_Performance')
                rolling_weights_df.to_excel(writer, sheet_name='Rolling_Weights')
            print("엑셀 파일 저장 완료.")
        except ImportError:
            print("\n⚠️ 'openpyxl' 라이브러리가 필요합니다. 'pip install openpyxl' 명령어로 설치해주세요.")
        except Exception as e:
            print(f"\n엑셀 파일 저장 중 오류 발생: {e}")

        plt.close('all')