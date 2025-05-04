# ---------------------- IMPORTS ----------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import base64
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ---------------------- CONFIG ----------------------
st.set_page_config(
    page_title="⚡ Gringotts' Wizardry: The Financial Forecasting Spellbook",
    layout="wide"
)

# ---------------------- BACKGROUND IMAGE ----------------------
try:
    with open("back.jpeg", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
except FileNotFoundError:
    encoded = ""
    st.warning("⚠️ Background image not found. Using default theme.")

# ---------------------- CUSTOM STYLES ----------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond&display=swap');
html, body {{
    background-image: url("data:image/jpeg;base64,{encoded}");
    background-size: cover;
    background-attachment: fixed;
    color: #f5deb3;
    font-family: 'EB Garamond', serif;
    cursor: url('https://cur.cursors-4u.net/cursors/cur-9/cur821.cur'), auto;
}}
h1, h2, h3, h4 {{
    color: gold;
    font-weight: bold;
    text-shadow: 1px 1px 2px black;
}}
.stDownloadButton > button, .stButton > button {{
    background-color: #4b0082;
    color: gold;
    font-weight: bold;
    border-radius: 8px;
}}
.stDownloadButton > button:hover, .stButton > button:hover {{
    background-color: gold;
    color: #4b0082;
    transform: scale(1.05);
    box-shadow: 0 0 10px gold;
}}
body::before {{
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    background: radial-gradient(circle, rgba(255,255,255,0.3) 1px, transparent 1px);
    background-size: 5px 5px;
    animation: sparkle 8s linear infinite;
    z-index: -1;
}}
@keyframes sparkle {{
    from {{ background-position: 0 0; }}
    to {{ background-position: 100% 100%; }}
}}
.blink-text {{
    font-size: 24px;
    font-weight: bold;
    color: gold;
    animation: blink 1s infinite;
    text-align: center;
}}
@keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0; }}
}}
</style>
""", unsafe_allow_html=True)

# ---------------------- APP TITLE ----------------------
st.markdown("<h1 style='text-align: center;'>⚡ Gringotts' Wizardry: The Financial Forecasting Spellbook</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------- SESSION STATE ----------------------
for key in ["df", "model", "X_train", "X_test", "y_train", "y_test", "y_pred"]:
    st.session_state.setdefault(key, None)

# ---------------------- SIDEBAR NAVIGATION ----------------------
PAGES = {
    "🏰 Gringotts Entrance Hall (Home)": "home",
    "📜 Summon Scrolls (Upload Data)": "upload",
    "🧽 Cast Scourgify (Preprocessing)": "preprocess",
    "🧪 Brew Features in Potions Class (Feature Engineering)": "features",
    "✂️ Divide with the Sword of Gryffindor (Train/Test Split)": "split",
    "🧙 Train the Magic Wand (Train Model)": "train",
    "🔮 Consult the Prophecy Orb (Evaluate)": "evaluate",
    "📜 Reveal the Seer's Predictions (Show Predictions)": "predict"
}

if "current_page" not in st.session_state:
    st.session_state.current_page = list(PAGES.keys())[0]

with st.sidebar:
    st.markdown("## ⚡ Choose Your Magic:")
    for page in PAGES:
        if st.button(page, key=page):
            st.session_state.current_page = page

page = st.session_state.current_page

# ---------------------- PAGE HANDLERS ----------------------

if page == "🏰 Gringotts Entrance Hall (Home)":
    st.markdown("<h2 style='text-align: center;'>Welcome to the Hogwarts School of Financial Wizardry!</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Use this magic-powered ML app to analyze stock data, forecast future values, and become a Financial Sorcerer!</p>", unsafe_allow_html=True)
    st.image("HP.gif", width=800)

elif page == "📜 Summon Scrolls (Upload Data)":
    st.subheader("📥 Upload a CSV file or Enter a Stock Ticker")
    ticker = st.text_input("🧙 Enter Stock Ticker (e.g., AAPL, GOOGL)")
    file = st.file_uploader("📜 Or Upload Your CSV File", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file)
            if df.empty or df.shape[1] < 2:
                st.error("❌ Uploaded file is empty or doesn't have enough columns.")
            else:
                st.session_state.df = df
                st.success("✨ Magic scroll (CSV) successfully uploaded!")
                st.dataframe(df.head())
                time.sleep(1)
                st.session_state.current_page = "🧽 Cast Scourgify (Preprocessing)"
                st.experimental_rerun()
        except Exception as e:
            st.error(f"🚫 Error: {e}")

    elif ticker:
        st.info("🔍 Summoning data from the enchanted Yahoo Finance archives...")
        try:
            data = yf.download(ticker, period="6mo")
            if data.empty:
                st.error("❌ No data found. Check ticker.")
            else:
                data.reset_index(inplace=True)
                st.session_state.df = data
                st.success(f"📈 {ticker.upper()} data successfully summoned!")
                st.dataframe(data.head())
                time.sleep(1)
                st.session_state.current_page = "🧽 Cast Scourgify (Preprocessing)"
                st.experimental_rerun()
        except Exception as e:
            st.error(f"⚠️ Failed to summon data: {e}")
    else:
        st.info("🧙 Please upload a scroll or enter a valid ticker.")

elif page == "🧽 Cast Scourgify (Preprocessing)":
    st.subheader("🧹 Clean the Data")
    df = st.session_state.df

    if df is not None:
        st.dataframe(df.head())
        st.write("### 📊 Summary", df.describe())
        st.write("### 🔍 Data Types", df.dtypes)
        st.write("### 🧟 Missing Values", df.isnull().sum()[df.isnull().sum() > 0])

        clean_method = st.radio("Choose cleaning method:", ["Drop missing rows", "Fill with forward fill", "Fill with zero", "Do nothing"])
        if clean_method == "Drop missing rows":
            df.dropna(inplace=True)
        elif clean_method == "Fill with forward fill":
            df.fillna(method='ffill', inplace=True)
        elif clean_method == "Fill with zero":
            df.fillna(0, inplace=True)

        st.session_state.df = df
        st.success("🧹 Data cleaned!")
        st.dataframe(df.head())
    else:
        st.warning("🧙 Please upload or summon data first.")

elif page == "🧪 Brew Features in Potions Class (Feature Engineering)":
    st.subheader("🧪 Feature Engineering Chamber")
    df = st.session_state.df

    if df is not None:
        st.dataframe(df.head())
        st.markdown("### 🧙 Choose Your Magic Spells (Feature Creation):")

        if "Close" in df.columns:
            ma_windows = st.multiselect("📈 Add Moving Averages for 'Close':", [5, 10, 20, 50, 100, 200], default=[5, 20])
            for window in ma_windows:
                df[f"MA_{window}"] = df["Close"].rolling(window=window).mean()

            if st.checkbox("📉 Add Daily % Change"):
                df["Pct_Change"] = df["Close"].pct_change()

        if st.checkbox("🧮 Add Custom Feature"):
            formula = st.text_input("Write a pandas-style formula (e.g., `High - Low`):")
            name = st.text_input("Name your new feature:")
            if st.button("🔬 Add Custom Feature"):
                try:
                    df[name] = df.eval(formula)
                    st.success(f"✨ Feature '{name}' created successfully!")
                except Exception as e:
                    st.error(f"⚠️ Error in formula: {e}")

        df.dropna(inplace=True)
        st.session_state.df = df
        st.success("🧪 Feature engineering complete!")
        st.dataframe(df.tail())
    else:
        st.warning("🧙 Please upload or summon data first.")

elif page == "🧙 Train the Magic Wand (Train Model)":
    st.subheader("🧪 Train Your Magical ML Model")
    df = st.session_state.df

    if df is not None:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("❌ Not enough numeric columns to train.")
        else:
            features = st.multiselect("📘 Select Features (X):", numeric_cols, default=numeric_cols[:-1])
            target = st.selectbox("🎯 Select Target (y):", numeric_cols, index=len(numeric_cols)-1)

            if features and target:
                test_size = st.slider("📏 Test set size (%):", 10, 50, 20)
                model_name = st.radio("🔮 Choose Model:", ["Linear Regression", "Decision Tree", "Random Forest"])

                if st.button("🧙 Cast Training Spell"):
                    try:
                        X = df[features]
                        y = df[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

                        model = {
                            "Linear Regression": LinearRegression(),
                            "Decision Tree": DecisionTreeRegressor(),
                            "Random Forest": RandomForestRegressor()
                        }[model_name]

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)

                        st.success(f"✨ {model_name} trained successfully!")
                        st.metric("📉 Mean Squared Error", round(mse, 2))

                        st.session_state.update({
                            "model": model,
                            "X_train": X_train,
                            "X_test": X_test,
                            "y_train": y_train,
                            "y_test": y_test,
                            "y_pred": y_pred
                        })

                        st.markdown("### 🔍 Prediction Preview")
                        preview = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).reset_index(drop=True)
                        st.dataframe(preview.head())

                    except Exception as e:
                        st.error(f"🚫 Training error: {e}")
            else:
                st.warning("🧙 Select both features and target.")
    else:
        st.warning("🧙 Upload and preprocess data first.")

    st.subheader("✂️ Divide the Data into Train and Test Sets")
    df = st.session_state.df

    if df is not None:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("❌ Not enough numeric columns to split.")
        else:
            features = st.multiselect("📘 Select Features (X):", numeric_cols, default=numeric_cols[:-1])
            target = st.selectbox("🎯 Select Target (y):", numeric_cols, index=len(numeric_cols)-1)

            if features and target:
                test_size = st.slider("📏 Test set size (%):", 10, 50, 20)
elif page == "✂️ Divide with the Sword of Gryffindor (Train/Test Split)":
    st.subheader("✂️ Divide the Data into Train and Test Sets")
    df = st.session_state.df

    if df is not None and not df.empty:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("❌ Not enough numeric columns to split.")
        else:
            features = st.multiselect("📘 Select Features (X):", numeric_cols, default=numeric_cols[:-1])
            target = st.selectbox("🎯 Select Target (y):", numeric_cols, index=len(numeric_cols)-1)

            if features and target:
                test_size = st.slider("📏 Test set size (%):", 10, 50, 20)

                if len(df[features]) > 1:  # Ensure enough data exists for train-test split
                    if st.button("✂️ Split Data"):
                        try:
                            X = df[features]
                            y = df[target]

                            # Ensure we have enough data to split
                            if len(X) < 2:  # Less than 2 rows can't be split
                                st.error("❌ Not enough data to perform a split.")
                            else:
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

                                st.success("✨ Data successfully split into Train and Test sets!")
                                st.session_state.update({
                                    "X_train": X_train,
                                    "X_test": X_test,
                                    "y_train": y_train,
                                    "y_test": y_test
                                })

                                st.write("### Train Set")
                                st.dataframe(X_train.head())

                                st.write("### Test Set")
                                st.dataframe(X_test.head())

                        except Exception as e:
                            st.error(f"🚫 Split error: {e}")
                else:
                    st.error("❌ Not enough rows to perform a train-test split. Please check the dataset.")
            else:
                st.warning("🧙 Select both features and target.")
    else:
        st.warning("🧙 Upload and preprocess data first.")

elif page == "🔮 Consult the Prophecy Orb (Evaluate)":
    st.subheader("🔮 Evaluate Your Model")
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred

    if model and X_test is not None:
        mse = mean_squared_error(y_test, y_pred)
        st.success(f"✨ Model Evaluation Complete!")
        st.metric("📉 Mean Squared Error", round(mse, 2))

        st.write("### Evaluation Metrics")
        st.write(f"🔮 Mean Squared Error: {round(mse, 2)}")

        st.write("### Predictions vs Actual")
        eval_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.dataframe(eval_df.head())
    else:
        st.warning("🧙 Please train the model first.")
elif page == "📜 Reveal the Seer's Predictions (Show Predictions)":
    st.subheader("📜 Show Predictions")
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred

    if model and X_test is not None:
        st.write("### Prediction Preview")
        predictions_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        }).reset_index(drop=True)

        st.dataframe(predictions_df)

        st.write("### Prediction Plot")
        st.line_chart(predictions_df.set_index("index"))
    else:
        st.warning("🧙 Please train and evaluate the model first.")
