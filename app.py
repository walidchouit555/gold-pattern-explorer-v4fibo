import requests
import streamlit as st
import pandas as pd
import numpy as np


# ==================================================
# 1. Page configuration
# ==================================================

st.set_page_config(
    page_title="Gold Pattern Explorer",
    layout="wide"
)

st.title("🟠 Gold Pattern Explorer")
st.write("Historical Gold Pattern Matching & Backtesting")
st.info(
    "يجلب التطبيق بيانات الذهب تلقائيًا عند الإمكان، مع الاحتفاظ بإمكانية "
    "رفع ملف CSV يدويًا كخيار بديل."
)

API_URL = "https://xaus.com/api/v1/chart"
MIN_HISTORY_MONTHS = 6
MAX_LATEST_CANDLE_AGE_HOURS = 72


# ==================================================
# 2. Data helpers
# ==================================================

def validate_history(dataframe):
    """Normalize OHLC history and reject invalid or insufficient data."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("لم تصل أي شموع صالحة من مصدر البيانات.")

    history = dataframe.copy()
    history.columns = [str(column).strip().lower() for column in history.columns]

    # Accept both CSV names and XAUS API's compact OHLCV names.
    history = history.rename(columns={
        "t": "timestamp",
        "time": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    })

    required_columns = ["timestamp", "open", "high", "low", "close"]
    missing_columns = [column for column in required_columns if column not in history.columns]
    if missing_columns:
        raise ValueError(
            "الأعمدة المطلوبة غير موجودة: " + ", ".join(missing_columns)
        )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"], utc=True, errors="coerce"
    )

    price_columns = ["open", "high", "low", "close"]
    for column in price_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    if "volume" in history.columns:
        history["volume"] = pd.to_numeric(history["volume"], errors="coerce")
    else:
        history["volume"] = np.nan

    history = history.dropna(subset=["timestamp"] + price_columns)
    history = history.sort_values("timestamp")
    history = history.drop_duplicates(subset=["timestamp"], keep="last")

    valid_ohlc = (
        (history["high"] >= history["low"])
        & (history["high"] >= history["open"])
        & (history["high"] >= history["close"])
        & (history["low"] <= history["open"])
        & (history["low"] <= history["close"])
        & (history[price_columns] > 0).all(axis=1)
    )
    history = history.loc[valid_ohlc].reset_index(drop=True)

    if history.empty:
        raise ValueError("لم تبقَ أي شموع صالحة بعد التحقق من الأسعار.")

    # Require at least six calendar months between the first and last candles.
    first_timestamp = history["timestamp"].min()
    last_timestamp = history["timestamp"].max()
    six_months_before_last = last_timestamp - pd.DateOffset(months=MIN_HISTORY_MONTHS)
    if first_timestamp > six_months_before_last:
        raise ValueError(
            "الفترة التاريخية المستلمة أقل من ستة أشهر. "
            f"الفترة المتاحة: {first_timestamp} — {last_timestamp}."
        )

    # The latest candle must not be excessively old. A 72-hour threshold allows
    # normal weekend/holiday gaps but rejects obviously stale automatic data.
    age = pd.Timestamp.now(tz="UTC") - last_timestamp
    if age > pd.Timedelta(hours=MAX_LATEST_CANDLE_AGE_HOURS):
        raise ValueError(
            "آخر شمعة قديمة جدًا للاستخدام التلقائي "
            f"(عمرها التقريبي {age.total_seconds() / 3600:.1f} ساعة)."
        )

    return history


def api_timestamp(value):
    """Parse XAUS timestamps expressed as Unix seconds/milliseconds or ISO text."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return pd.NaT
        unit = "ms" if abs(float(value)) >= 1e12 else "s"
        return pd.to_datetime(value, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def fetch_automatic_history():
    """Fetch two years of hourly XAU/USD candles and validate the response."""
    response = requests.get(
        API_URL,
        params={"symbol": "xau", "range": "2y", "interval": "1h"},
        timeout=25,
        headers={"User-Agent": "GoldPatternExplorer/1.0"},
    )
    response.raise_for_status()
    payload = response.json()

    data_state = payload.get("data_state", {}) if isinstance(payload, dict) else {}
    api_status = str(data_state.get("status", "")).lower() if isinstance(data_state, dict) else ""
    if api_status == "unavailable":
        raise ValueError("مصدر البيانات يعلن أن البيانات غير متاحة حاليًا.")
    if api_status == "stale":
        raise ValueError("مصدر البيانات يعلن أن البيانات التاريخية قديمة؛ لم تُستخدم تلقائيًا.")

    if isinstance(payload, list):
        points = payload
    elif isinstance(payload, dict):
        points = None
        for key in ("points", "data", "candles", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                points = value
                break
        if points is None:
            # Handle a standard chart-response wrapper if the upstream changes shape.
            chart = payload.get("chart")
            if isinstance(chart, dict):
                result = chart.get("result")
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    timestamps = result[0].get("timestamp", [])
                    quote = (result[0].get("indicators", {}).get("quote") or [{}])[0]
                    points = [
                        {
                            "t": timestamps[i],
                            "o": (quote.get("open") or [None] * len(timestamps))[i],
                            "h": (quote.get("high") or [None] * len(timestamps))[i],
                            "l": (quote.get("low") or [None] * len(timestamps))[i],
                            "c": (quote.get("close") or [None] * len(timestamps))[i],
                            "v": (quote.get("volume") or [None] * len(timestamps))[i],
                        }
                        for i in range(len(timestamps))
                    ]
        if points is None:
            raise ValueError("صيغة استجابة مصدر البيانات غير معروفة.")
    else:
        raise ValueError("استجابة مصدر البيانات ليست بصيغة JSON المتوقعة.")

    if not points:
        raise ValueError("مصدر البيانات لم يُرجع أي شموع.")

    raw = pd.DataFrame(points)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    timestamp_column = next(
        (column for column in ("t", "timestamp", "time", "datetime", "date") if column in raw.columns),
        None,
    )
    if timestamp_column is None:
        raise ValueError("استجابة مصدر البيانات لا تحتوي على وقت الشمعة.")

    raw["timestamp"] = raw[timestamp_column].map(api_timestamp)
    raw = raw.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    if timestamp_column != "timestamp" and timestamp_column in raw.columns:
        # The parsed timestamp column above is authoritative.
        raw = raw.drop(columns=[timestamp_column])

    history = validate_history(raw)
    return history, data_state


def load_csv_history(uploaded_file):
    """Read an uploaded CSV using the same validation rules as automatic data."""
    csv_data = pd.read_csv(uploaded_file)
    return validate_history(csv_data)


# ==================================================
# 3. Automatic data fetching, with manual fallback
# ==================================================

if "auto_fetch_attempted" not in st.session_state:
    st.session_state["auto_fetch_attempted"] = True
    try:
        with st.spinner("جارٍ جلب بيانات الذهب التاريخية تلقائيًا..."):
            auto_history, auto_state = fetch_automatic_history()
        st.session_state["auto_history_data"] = auto_history
        st.session_state["auto_data_state"] = auto_state
        st.session_state["auto_fetch_error"] = None
        st.session_state["source_mode"] = "Automatic data"
    except Exception as error:
        st.session_state["auto_history_data"] = None
        st.session_state["auto_data_state"] = {}
        st.session_state["auto_fetch_error"] = str(error)
        st.session_state["source_mode"] = "Upload CSV"

st.header("1. Data Source")

if st.button("🔄 Refresh automatic gold data"):
    try:
        with st.spinner("جارٍ تحديث بيانات الذهب التاريخية..."):
            auto_history, auto_state = fetch_automatic_history()
        st.session_state["auto_history_data"] = auto_history
        st.session_state["auto_data_state"] = auto_state
        st.session_state["auto_fetch_error"] = None
        st.session_state["source_mode"] = "Automatic data"
        st.success("تم تحديث البيانات التلقائية بنجاح.")
    except Exception as error:
        st.session_state["auto_fetch_error"] = str(error)
        st.warning(
            "تعذّر تحديث البيانات التلقائية. يمكنك متابعة العمل بملف CSV اليدوي. "
            f"التفاصيل: {error}"
        )

source_mode = st.radio(
    "Choose data source",
    options=["Automatic data", "Upload CSV"],
    horizontal=True,
    key="source_mode",
)

history = None

if source_mode == "Automatic data":
    auto_history = st.session_state.get("auto_history_data")
    if isinstance(auto_history, pd.DataFrame) and not auto_history.empty:
        history = auto_history.copy()
        st.success(
            f"تم تحميل {len(history):,} شمعة تاريخية تلقائيًا. "
            "المصدر: XAUS (XAU/USD، إطار الساعة)."
        )
        st.caption(
            "الفترة التاريخية (UTC): "
            f"{history['timestamp'].min()} — {history['timestamp'].max()}"
        )
        state = st.session_state.get("auto_data_state", {})
        if isinstance(state, dict) and state.get("as_of"):
            st.caption(f"حالة/وقت المصدر حسب API: {state.get('status', 'غير محدد')} — {state['as_of']}")
        st.dataframe(history.tail(5), use_container_width=True, hide_index=True)
    else:
        error_message = st.session_state.get("auto_fetch_error")
        st.warning(
            "البيانات التلقائية غير متاحة حاليًا. اختر «Upload CSV» "
            "للعودة إلى الطريقة القديمة، أو اضغط زر التحديث لإعادة المحاولة."
        )
        if error_message:
            st.caption(f"تفاصيل فشل الجلب التلقائي: {error_message}")
else:
    history_file = st.file_uploader(
        "Upload historical gold data (CSV)",
        type=["csv"],
        key="history",
        help="يجب أن يحتوي الملف على timestamp, open, high, low, close."
    )
    if history_file is not None:
        try:
            history = load_csv_history(history_file)
            st.session_state["csv_history_data"] = history
            st.success(f"Loaded {len(history):,} valid candles from CSV.")
            st.caption(
                "الفترة التاريخية (UTC): "
                f"{history['timestamp'].min()} — {history['timestamp'].max()}"
            )
            st.dataframe(history.tail(5), use_container_width=True, hide_index=True)
        except Exception as error:
            st.error(f"تعذرت قراءة ملف CSV أو التحقق منه: {error}")
    elif isinstance(st.session_state.get("csv_history_data"), pd.DataFrame):
        history = st.session_state["csv_history_data"].copy()
        st.info("تم الاحتفاظ بملف CSV الذي رفعته سابقًا خلال هذه الجلسة.")

if isinstance(history, pd.DataFrame):
    st.session_state["history_data"] = history.copy()
else:
    st.session_state.pop("history_data", None)


# ==================================================
# 4. Historical pattern matching (existing method retained)
# ==================================================

st.divider()
st.header("2. Historical Matching")

if "history_data" not in st.session_state:
    st.info("اختر مصدر البيانات وحمّل البيانات قبل بدء البحث عن الأنماط.")
else:
    history = st.session_state["history_data"].copy()

    pattern_length = st.selectbox(
        "Pattern length (hourly candles)",
        [48, 72, 96, 144],
        index=3,
        key="matching_pattern_length"
    )

    # Changing the source or pattern length should not reuse an old target
    # index that would move the target away from the latest candles.
    active_source = st.session_state.get("_active_data_source")
    previous_pattern_length = st.session_state.get("_previous_pattern_length")
    if active_source != source_mode or (
        previous_pattern_length is not None
        and previous_pattern_length != pattern_length
    ):
        for length in [48, 72, 96, 144]:
            st.session_state.pop(f"matching_target_start_{length}", None)
    st.session_state["_active_data_source"] = source_mode
    st.session_state["_previous_pattern_length"] = pattern_length

    minimum_candles = pattern_length * 2 + 1

    if len(history) < minimum_candles:
        st.warning(
            f"At least {minimum_candles} valid candles are needed "
            f"for a {pattern_length}-candle pattern and an earlier historical comparison."
        )
    else:
        max_start = len(history) - pattern_length
        target_start = st.number_input(
            "Target starting candle index",
            min_value=0,
            max_value=max_start,
            value=max_start,
            step=1,
            key=f"matching_target_start_{pattern_length}"
        )

        target_start = int(target_start)
        target_end = target_start + pattern_length
        target = history.iloc[target_start:target_end].copy()

        st.subheader("Selected target")
        st.write(
            "Target period (UTC):",
            str(target["timestamp"].iloc[0]),
            "—",
            str(target["timestamp"].iloc[-1])
        )
        st.caption(
            "عند اختيار آخر فهرس، تكون العينة هي أحدث الشموع المتاحة "
            "بعدد يساوي طول النمط المختار."
        )
        st.line_chart(target.set_index("timestamp")["close"])

        if len(target) != pattern_length:
            st.warning("The selected target does not contain the required number of candles.")
        else:
            target_close = target["close"].to_numpy(dtype=float)
            target_base = target_close[0]

            if not np.isfinite(target_close).all() or target_base == 0:
                st.error("The target pattern contains invalid prices.")
            else:
                target_shape = (target_close / target_base) - 1
                results = []

                # Historical candidates end before the selected target begins.
                for start in range(0, target_start - pattern_length + 1):
                    end = start + pattern_length
                    candidate = history.iloc[start:end]
                    if len(candidate) != pattern_length:
                        continue

                    candidate_close = candidate["close"].to_numpy(dtype=float)
                    if not np.isfinite(candidate_close).all():
                        continue
                    candidate_base = candidate_close[0]
                    if candidate_base == 0:
                        continue

                    candidate_shape = (candidate_close / candidate_base) - 1
                    rmse = float(np.sqrt(np.mean((target_shape - candidate_shape) ** 2)))
                    similarity = 100 / (1 + rmse * 100)
                    future = history.iloc[end:min(end + 12, len(history))].copy()

                    results.append({
                        "start": start,
                        "end": end,
                        "rmse": rmse,
                        "similarity": similarity,
                        "future": future,
                    })

                results.sort(key=lambda item: item["rmse"])
                st.subheader("Historical matching results")

                if not results:
                    st.warning("No eligible historical matches were found. Try another target starting index.")
                else:
                    st.success(f"Found {len(results):,} historical matches.")
                    top_n = st.slider(
                        "Number of matches to display",
                        min_value=1,
                        max_value=min(20, len(results)),
                        value=min(10, len(results))
                    )

                    summary = []
                    for rank, result in enumerate(results[:top_n], start=1):
                        matched = history.iloc[result["start"]:result["end"]]
                        summary.append({
                            "Rank": rank,
                            "Similarity (%)": round(result["similarity"], 2),
                            "Start (UTC)": str(matched["timestamp"].iloc[0]),
                            "End (UTC)": str(matched["timestamp"].iloc[-1]),
                        })

                    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
                    st.caption(
                        "Similarity is a mathematical shape score, not a probability or a price forecast."
                    )

                    selected_rank = st.selectbox(
                        "Inspect a historical match",
                        options=list(range(1, top_n + 1)),
                        format_func=lambda rank: f"Match {rank}"
                    )
                    selected = results[selected_rank - 1]
                    st.write("Similarity score:", f"{selected['similarity']:.2f}%")

                    matched = history.iloc[selected["start"]:selected["end"]].copy()
                    st.subheader("Matched historical pattern")
                    st.line_chart(matched.set_index("timestamp")["close"])

                    future = selected["future"]
                    if not future.empty:
                        st.subheader("What happened afterward")
                        st.line_chart(future.set_index("timestamp")["close"])
                        st.dataframe(
                            future[["timestamp", "open", "high", "low", "close"]],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No subsequent candles are available after this historical match.")

# ==================================================
# 5. Volatility Breakout Strategy + historical backtest
# ==================================================

st.divider()
st.header("3. Volatility Breakout Strategy — configurable Fibonacci")
st.write(
    "تبحث الاستراتيجية عن شمعة يتجاوز مداها (High − Low) أكبر مدى في الشموع "
    "السابقة، ثم تنتظر رجوع السعر إلى مستوى تصحيح فيبوناتشي الذي تختاره، "
    "وتغلق عند امتداد فيبوناتشي الذي تحدده كهدف."
)

with st.expander("Strategy settings", expanded=True):
    lookback_hours = st.number_input(
        "Reference window: previous hourly candles",
        min_value=24, max_value=168, value=24, step=1,
        help="تُقارن كل شمعة بالشموع السابقة فقط، ولا تدخل ضمن نافذتها المرجعية."
    )
    fib_retracement_levels = [23.6, 38.2, 50.0, 61.8, 78.6]
    entry_fib = st.selectbox(
        "Entry — Fibonacci Retracement",
        options=fib_retracement_levels,
        index=0,
        format_func=lambda value: f"{value:g}%",
        help="اختر مستوى التصحيح الذي ينتظر التطبيق وصول السعر إليه للدخول."
    )
    fib_extension_levels = [100.0, 127.2, 141.0, 161.8, 200.0, 261.8]
    target_extension = st.selectbox(
        "Final target — Fibonacci Extension",
        options=fib_extension_levels,
        index=3,
        format_func=lambda value: f"{value:g}%",
        help="اختر امتداد فيبوناتشي الذي سيكون هدف إغلاق الصفقة."
    )
    col_a, col_b = st.columns(2)
    col_a.metric("Entry retracement", f"{entry_fib:g}%")
    col_b.metric("Final target extension", f"{target_extension:g}%")
    max_wait = st.number_input(
        "Maximum waiting candles after signal (0 = unlimited)",
        min_value=0, max_value=50000, value=0, step=1,
        help=(
            "الصفر يعني الانتظار دون حد زمني. أي إشارة أحدث تستبدل الإشارة "
            "المنتظرة القديمة. لا يمكن استخدام الإشارة نفسها لفتح أكثر من صفقة."
        )
    )
    pending_mode = st.selectbox(
        "New signals while a trade is open",
        options=[
            "Keep only the newest signal pending",
            "Ignore signals while a trade is open",
        ],
        index=0,
        help=(
            "الخيار الافتراضي يحتفظ بآخر إشارة فقط حتى تُغلق الصفقة الحالية؛ "
            "ولا يسمح بفتح صفقتين متزامنتين."
        )
    )

if "history_data" not in st.session_state:
    st.info("اختر مصدر البيانات في الأعلى أولًا.")
else:
    strat_history = st.session_state["history_data"].copy().reset_index(drop=True)
    strat_history["range"] = strat_history["high"] - strat_history["low"]
    strat_history["previous_max_range"] = (
        strat_history["range"].shift(1).rolling(int(lookback_hours)).max()
    )
    strat_history["signal"] = (
        strat_history["previous_max_range"].notna()
        & (strat_history["range"] > strat_history["previous_max_range"])
        & (strat_history["range"] > 0)
    )

    signal_rows = strat_history[strat_history["signal"]]
    if len(strat_history) <= int(lookback_hours) + 2:
        st.warning("لا توجد شموع كافية لتشغيل الاستراتيجية.")
    else:
        st.caption(
            f"عدد الشموع: {len(strat_history):,} | "
            f"الفترة: {strat_history['timestamp'].iloc[0]} — "
            f"{strat_history['timestamp'].iloc[-1]}"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Exceptional candles", f"{len(signal_rows):,}")
        col2.metric("Reference hours", int(lookback_hours))
        col3.metric("Entry / Final target", f"{entry_fib:g}% / {target_extension:g}%")

        if not signal_rows.empty:
            display_signals = signal_rows[
                ["timestamp", "open", "high", "low", "close", "range", "previous_max_range"]
            ].copy()
            display_signals["range"] = display_signals["range"].round(5)
            display_signals["previous_max_range"] = display_signals["previous_max_range"].round(5)
            st.subheader("Detected exceptional candles")
            st.dataframe(display_signals.tail(200), use_container_width=True, hide_index=True)
            st.caption("يعرض الجدول أحدث 200 إشارة كحد أقصى.")

        st.subheader("Walk-forward backtest")
        st.caption(
            "قواعد الاختبار: لا دخول في شمعة الإشارة نفسها؛ وقف الخسارة ثابت عند الطرف "
            "المقابل لشمعة الإشارة؛ إذا لُمِس الوقف والهدف داخل شمعة واحدة يُحتسب الوقف أولًا. "
            "لا توجد قاعدة 68% ولا تتبّع لوقف الخسارة. النتائج لا تشمل السبريد أو العمولة أو الانزلاق."
        )

        trades = []
        pending_signal = None
        open_trade = None
        replaced_signals = 0
        invalidated_signals = 0
        expired_signals = 0

        def make_signal(signal_index, signal_row):
            high = float(signal_row["high"])
            low = float(signal_row["low"])
            span = high - low
            bullish = float(signal_row["close"]) >= float(signal_row["open"])
            if span <= 0:
                return None
            if bullish:
                entry = high - span * (entry_fib / 100.0)
                initial_stop = low
                target = low + span * (target_extension / 100.0)
            else:
                entry = low + span * (entry_fib / 100.0)
                initial_stop = high
                target = high - span * (target_extension / 100.0)
            return {
                "signal_index": int(signal_index),
                "signal_time": signal_row["timestamp"],
                "direction": "Buy" if bullish else "Sell",
                "bullish": bool(bullish),
                "signal_high": high,
                "signal_low": low,
                "entry": float(entry),
                "initial_stop": float(initial_stop),
                "target": float(target),
            }

        def finish_trade(trade, exit_index, exit_price, exit_reason):
            reward = (float(exit_price) - trade["entry"]) if trade["bullish"] else (
                trade["entry"] - float(exit_price)
            )
            risk = abs(trade["entry"] - trade["initial_stop"])
            trades.append({
                "Signal time (UTC)": trade["signal_time"],
                "Entry time (UTC)": strat_history.iloc[trade["entry_index"]]["timestamp"],
                "Exit time (UTC)": strat_history.iloc[exit_index]["timestamp"],
                "Direction": trade["direction"],
                "Entry": trade["entry"],
                "Initial stop": trade["initial_stop"],
                "Stop at exit": trade["current_stop"],
                f"Target {target_extension:g}% extension": trade["target"],
                "Exit price": float(exit_price),
                "Exit reason": exit_reason,
                "Result (R)": reward / risk if risk > 0 else np.nan,
                "Result (%) of entry price": (reward / trade["entry"] * 100.0)
                    if trade["entry"] else np.nan,
            })

        for i in range(len(strat_history)):
            candle = strat_history.iloc[i]
            candle_open = float(candle["open"])
            candle_high = float(candle["high"])
            candle_low = float(candle["low"])
            had_open_at_start = open_trade is not None

            # 1) Manage a trade that was already open before this candle.
            if open_trade is not None and i > open_trade["entry_index"]:
                trade = open_trade
                bullish = trade["bullish"]
                current_stop = trade["current_stop"]
                if bullish:
                    hit_stop = candle_low <= current_stop
                    hit_target = candle_high >= trade["target"]
                else:
                    hit_stop = candle_high >= current_stop
                    hit_target = candle_low <= trade["target"]

                if hit_stop:
                    # If an hourly candle gaps through the stop, model the worse opening price.
                    if bullish and candle_open < current_stop:
                        fill_price = candle_open
                    elif (not bullish) and candle_open > current_stop:
                        fill_price = candle_open
                    else:
                        fill_price = current_stop
                    finish_trade(trade, i, fill_price, trade["current_stop_reason"])
                    open_trade = None
                elif hit_target:
                    finish_trade(trade, i, trade["target"], f"Take profit {target_extension:g}% extension")
                    open_trade = None

            # 2) Validate a pending signal even while another trade is open.
            # A pending setup is invalid once its original stop is crossed or its wait expires.
            if pending_signal is not None and i > pending_signal["signal_index"]:
                age = i - pending_signal["signal_index"]
                if int(max_wait) > 0 and age > int(max_wait):
                    pending_signal = None
                    expired_signals += 1
                elif (
                    (pending_signal["bullish"] and candle_low <= pending_signal["initial_stop"])
                    or ((not pending_signal["bullish"]) and candle_high >= pending_signal["initial_stop"])
                ):
                    pending_signal = None
                    invalidated_signals += 1

            # 3) Only attempt an entry if no position was open at the START of this candle.
            # If a trade closed during this candle, wait until the next candle to avoid guessing
            # whether the exit happened before or after a pending entry touch.
            if not had_open_at_start and open_trade is None and pending_signal is not None:
                if i > pending_signal["signal_index"]:
                    touched_entry = candle_low <= pending_signal["entry"] <= candle_high
                    stop_crossed = (
                        (pending_signal["bullish"] and candle_low <= pending_signal["initial_stop"])
                        or ((not pending_signal["bullish"]) and candle_high >= pending_signal["initial_stop"])
                    )
                    if stop_crossed:
                        pending_signal = None
                        invalidated_signals += 1
                    elif touched_entry:
                        new_trade = dict(pending_signal)
                        new_trade["entry_index"] = int(i)
                        new_trade["current_stop"] = new_trade["initial_stop"]
                        new_trade["current_stop_reason"] = "Initial stop loss"
                        open_trade = new_trade
                        pending_signal = None

            # 4) Signal is confirmed at the candle close, so it cannot enter on that same candle.
            if bool(candle["signal"]):
                new_signal = make_signal(i, candle)
                if new_signal is not None:
                    if open_trade is not None:
                        if pending_mode == "Keep only the newest signal pending":
                            if pending_signal is not None:
                                replaced_signals += 1
                            pending_signal = new_signal
                    else:
                        if pending_signal is not None:
                            replaced_signals += 1
                        pending_signal = new_signal

        # If a position is still open at the last candle, mark it to market but exclude it
        # from closed-trade performance statistics.
        if open_trade is not None:
            last_index = len(strat_history) - 1
            last_close = float(strat_history.iloc[last_index]["close"])
            finish_trade(open_trade, last_index, last_close, "Open at end of data (marked to market)")

        trade_df = pd.DataFrame(trades)
        closed = pd.DataFrame()
        if not trade_df.empty:
            closed = trade_df[
                trade_df["Exit reason"] != "Open at end of data (marked to market)"
            ].copy()

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Closed trades", f"{len(closed):,}")
        col_b.metric("Open at end", "1" if (
            not trade_df.empty and (trade_df["Exit reason"] == "Open at end of data (marked to market)").any()
        ) else "0")
        col_c.metric("Signals replaced", f"{replaced_signals:,}")

        if trade_df.empty:
            st.info("لم تُنتج البيانات صفقات دخول وفق الشروط الحالية.")
        else:
            wins = int((closed["Result (R)"] > 0).sum()) if not closed.empty else 0
            losses = int((closed["Result (R)"] < 0).sum()) if not closed.empty else 0
            breakeven = int((closed["Result (R)"].abs() <= 1e-10).sum()) if not closed.empty else 0
            total_closed = len(closed)
            win_rate = wins / total_closed * 100.0 if total_closed else 0.0
            total_r = float(closed["Result (R)"].sum()) if total_closed else 0.0
            avg_r = float(closed["Result (R)"].mean()) if total_closed else 0.0
            gross_profit = float(closed.loc[closed["Result (R)"] > 0, "Result (R)"].sum()) if total_closed else 0.0
            gross_loss = float(-closed.loc[closed["Result (R)"] < 0, "Result (R)"].sum()) if total_closed else 0.0
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0
                else (float("inf") if gross_profit > 0 else float("nan"))
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Win rate (closed trades)", f"{win_rate:.2f}%")
            m2.metric("Net result", f"{total_r:.2f} R")
            m3.metric("Average result", f"{avg_r:.2f} R")
            m4.metric(
                "Profit factor",
                "∞" if np.isinf(profit_factor) else (
                    "—" if pd.isna(profit_factor) else f"{profit_factor:.2f}"
                )
            )
            st.dataframe(trade_df.round(5), use_container_width=True, hide_index=True)
            st.download_button(
                "Download backtest trades (CSV)",
                data=trade_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="volatility_breakout_backtest_v2.csv",
                mime="text/csv",
            )
            if total_closed:
                st.write(
                    f"الرابحة: {wins} | الخاسرة: {losses} | "
                    f"التعادل: {breakeven} | متوسط النتيجة: {avg_r:.2f}R"
                )

        if pending_signal is not None:
            waited = (len(strat_history) - 1) - pending_signal["signal_index"]
            st.info(
                f"هناك إشارة أخيرة ما زالت تنتظر الدخول: {pending_signal['direction']}، "
                f"من {pending_signal['signal_time']}، مستوى الدخول {pending_signal['entry']:.5f}. "
                f"عدد الشموع المنقضية منذ الإشارة: {waited}."
            )
        st.caption(
            f"إشارات أُلغيت بسبب تجاوز الوقف الأصلي: {invalidated_signals:,} | "
            f"إشارات انتهى وقت انتظارها: {expired_signals:,}. "
            "إذا كان الانتظار = 0، فلا تنتهي الإشارة بسبب الوقت؛ لكن إشارة أحدث قد تستبدلها."
        )
        st.warning(
            "هذا اختبار تاريخي افتراضي، وليس ضمانًا للربح أو توصية بالتداول. "
            "الشموع ذات إطار الساعة لا تكشف ترتيب السعر داخل الشمعة؛ لذلك استُخدمت "
            "افتراضات محافظة، ولم تُحتسب تكاليف التداول."
        )
