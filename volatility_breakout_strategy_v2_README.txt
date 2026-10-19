Gold Pattern Explorer — Configurable Fibonacci Strategy

Included files
- app.py: separate version of the app. Existing automatic data-loading logic and Historical Matching section are retained.
- requirements.txt: required Python packages.

Deploy safely
1. Keep your current working repository unchanged as a backup.
2. Create a NEW GitHub repository for this version.
3. Upload app.py and requirements.txt from this ZIP to the repository root.
4. Deploy the new repository on Streamlit Community Cloud and choose app.py.
5. Verify automatic data loading and Historical Matching before evaluating the strategy. Do not delete the old app.

Data and Historical Matching
- Automatic XAU/USD hourly history uses the existing XAUS endpoint.
- CSV upload remains available as a fallback.
- The existing Historical Matching section is retained; only the strategy/backtest section was changed.

Strategy rules
- Signal: the current hourly candle's High-Low range must be strictly greater than the maximum range of the preceding reference candles (24 by default; configurable).
- Direction: a candle closing at or above its open creates a Buy signal; a red candle creates a Sell signal.
- Entry: choose from Fibonacci retracement levels 23.6%, 38.2%, 50%, 61.8%, or 78.6% in the app.
- Initial stop: fixed at the opposite extreme of the signal candle.
- Final target: choose from Fibonacci extensions 100%, 127.2%, 141%, 161.8%, 200%, or 261.8% in the app.
- No 68% adverse-move rule, no break-even rule, and no trailing stop. The initial stop remains unchanged for the entire trade.
- Maximum waiting candles after signal: 0 means unlimited. Any newer signal replaces an older pending signal.
- Each signal can open at most one trade. If price crosses the original stop before entry, the pending setup is invalidated.
- Only one position can be open at a time. By default, the newest signal during an open trade is kept pending until the position closes; an optional setting lets you ignore signals while a trade is open.
- No entry is allowed on the signal candle. If a trade closes inside an hourly candle, a pending setup is not allowed to enter until the next candle.
- If the active stop and target are both touched within one hourly candle, the stop is assumed to have been touched first (conservative assumption).

Important limitations
- Historical results do not guarantee future performance and are not a recommendation to trade.
- Spread, commissions, and slippage are not included.
- Hourly OHLC data cannot reveal the exact sequence of price movement inside a candle.
- Open positions at the end of the dataset are marked to market in the trade table but excluded from closed-trade performance statistics.
