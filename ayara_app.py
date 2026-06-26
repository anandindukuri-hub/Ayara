import os
import io
import json
import time
import threading
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except Exception:
    GSPREAD_AVAILABLE = False



BASE_DIR = "Resources"
LOGO_PATH = f"{BASE_DIR}/Ayara.png"
INFLATION_IMG_PATH = f"{BASE_DIR}/Inflation.jpg"
DATA_PATH = f"{BASE_DIR}/Revised Final Dataset.csv"
MODELS_DIR = f"{BASE_DIR}/Time Series Models"


GEMINI_MODEL = "gemini-2.5-flash"


MAX_REQUESTS_PER_MINUTE = 8     
MAX_REQUESTS_PER_DAY = 200      


MAX_UPLOAD_MB = 7               
MAX_FILES_PER_RUN = 8           


USAGE_FILE = os.path.join(os.path.expanduser("~"), ".ayara_gemini_usage.json")

PRODUCT_LIST = ["Fruits", "Vegetables", "Spices", "Fuel and light", "Health"]
SECTORS = ["Urban", "Rural", "Rural+Urban"]

MODEL_PATHS = {
    ("Vegetables", "Urban"): f"{MODELS_DIR}/vegetables-urban.pkl",
    ("Vegetables", "Rural"): f"{MODELS_DIR}/vegetables-rural.pkl",
    ("Vegetables", "Rural+Urban"): f"{MODELS_DIR}/Vegetables-Total.pkl",
    ("Fuel and light", "Urban"): f"{MODELS_DIR}/Fuel and light-urban.pkl",
    ("Fuel and light", "Rural"): f"{MODELS_DIR}/Fuel and light-rural.pkl",
    ("Fuel and light", "Rural+Urban"): f"{MODELS_DIR}/Fuel and light-total.pkl",
    ("Health", "Urban"): f"{MODELS_DIR}/Health-urban.pkl",
    ("Health", "Rural"): f"{MODELS_DIR}/Health-rural.pkl",
    ("Health", "Rural+Urban"): f"{MODELS_DIR}/Health-Total.pkl",
    ("Fruits", "Urban"): f"{MODELS_DIR}/Fruits-urban.pkl",
    ("Fruits", "Rural"): f"{MODELS_DIR}/Fruits-Rural.pkl",
    ("Fruits", "Rural+Urban"): f"{MODELS_DIR}/Fruits-Total.pkl",
    ("Spices", "Urban"): f"{MODELS_DIR}/Spices-Urban.pkl",
    ("Spices", "Rural"): f"{MODELS_DIR}/spices-rural.pkl",
    ("Spices", "Rural+Urban"): f"{MODELS_DIR}/spices-total.pkl",
}



st.set_page_config(layout="wide", page_title="Ayara")



def safe_image(path, caption=None):
    
    if os.path.exists(path):
        st.image(path, caption=caption)
    else:
        st.info(f"Image not found at: {path}\n\nUpdate the path in the CONFIG block to display it.")


@st.cache_data(show_spinner=False)
def load_data(path):
    
    data = pd.read_csv(path)
    data["Month_Year"] = pd.to_datetime(data["Month_Year"])
    return data


def get_data():
    
    try:
        return load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Could not find the dataset at: {DATA_PATH}\n\nUpdate DATA_PATH in the CONFIG block.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load the dataset: {e}")
        st.stop()


def get_api_key():
    
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
    
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def key_looks_valid(key):
    
    if not key or not isinstance(key, str):
        return False
    k = key.strip()
    if "InsertAPIKEY" in k or k.lower() == "insertapikeyhere":
        return False
    return k.startswith("AIza") and len(k) >= 30


def ai_ready():
    
    return GEMINI_AVAILABLE and key_looks_valid(get_api_key())



@st.cache_resource
def _rate_state():
    
    return {"minute_times": [], "lock": threading.Lock()}


def _pacific_today():
    
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        
        return (datetime.utcnow() - timedelta(hours=8)).strftime("%Y-%m-%d")


def _read_day_usage():
    
    try:
        with open(USAGE_FILE, "r") as f:
            data = json.load(f)
        if data.get("date") == _pacific_today():
            return int(data.get("count", 0))
    except Exception:
        pass
    return 0


def _write_day_usage(count):
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump({"date": _pacific_today(), "count": int(count)}, f)
    except Exception:
        pass  

def reserve_quota():
 
    state = _rate_state()
    now = time.time()
    with state["lock"]:
        
        recent = [t for t in state["minute_times"] if now - t < 60]
        if len(recent) >= MAX_REQUESTS_PER_MINUTE:
            wait = int(60 - (now - min(recent))) + 1
            state["minute_times"] = recent
            return False, (
                f"Per-minute safety limit reached ({MAX_REQUESTS_PER_MINUTE}/min). "
                f"Please wait about {wait} seconds and try again."
            )
        
        day_count = _read_day_usage()
        if day_count >= MAX_REQUESTS_PER_DAY:
            return False, (
                f"Daily safety limit reached ({MAX_REQUESTS_PER_DAY} requests today). "
                
            )

        recent.append(now)
        state["minute_times"] = recent
        _write_day_usage(day_count + 1)
        return True, ""


def quota_status():
   
    return _read_day_usage(), MAX_REQUESTS_PER_DAY



def _gemini_generate(contents, response_json=False, max_retries=3):
   
    if not GEMINI_AVAILABLE:
        return "__ERROR__The google-genai package is not installed. Run: pip install google-genai"
    key = get_api_key()
    if not key_looks_valid(key):
        return "__ERROR__No valid Gemini API key found. Add GEMINI_API_KEY to .streamlit/secrets.toml"

    allowed, reason = reserve_quota()
    if not allowed:
        return f"__ERROR__{reason}"

    config = None
    if response_json:
        try:
            config = genai_types.GenerateContentConfig(response_mime_type="application/json")
        except Exception:
            config = None

    last_err = ""
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=key)
            if config is not None:
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=contents, config=config
                )
            else:
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=contents
                )
            text = getattr(resp, "text", None)
            if text:
                return text.strip()
            return "__ERROR__The model returned an empty response. Please try again."
        except Exception as e:
            
            msg = str(e)
            transient = any(code in msg for code in ("429", "503", "500", "deadline", "timeout"))
            last_err = "rate limit or temporary server error" if transient else "request failed"
            if transient and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  
                continue
            break
    return f"__ERROR__Gemini {last_err}. Please try again in a moment."


@st.cache_data(show_spinner=False)
def ai_translate_simplify(text, language, simplify):
    
    if not GEMINI_AVAILABLE or not key_looks_valid(get_api_key()):
        return None
    if simplify:
        instruction = (
            "Rewrite the inflation analysis below so it is noticeably shorter, clearer, and easy "
            "for a general reader to understand. Keep every important fact, number, and recommendation, "
            "remove repetition and jargon, and use plain, friendly language. "
            f"Write the final result in {language}."
        )
    else:
        instruction = (
            f"Translate the inflation analysis below into {language}. "
            "Preserve all facts, numbers, structure, and meaning. Do not add any commentary."
        )
    prompt = (
        f"{instruction}\n\n"
        "Return ONLY the rewritten/translated text, with no preamble or explanation.\n\n"
        f"---\n{text}"
    )
    return _gemini_generate(prompt)


@st.cache_data(show_spinner=False)
def ocr_receipt(file_bytes, mime_type):
    
    if not GEMINI_AVAILABLE or not key_looks_valid(get_api_key()):
        return {"__error__": "Gemini is not configured. Add GEMINI_API_KEY to enable receipt reading."}

    prompt = (
        "You are a receipt parser. Read this receipt and return ONLY JSON (no markdown, "
        "no explanation) with exactly these fields:\n"
        "{\n"
        '  "merchant_name": string or null,\n'
        '  "transaction_date": "YYYY-MM-DD" or null,\n'
        '  "currency": string or null,\n'
        '  "total": number or null,\n'
        '  "tax": number or null,\n'
        '  "items": [ { "description": string, "quantity": number, '
        '"unit_price": number, "line_total": number } ]\n'
        "}\n"
        "Normalize dates to YYYY-MM-DD. If a value is missing, use null. "
        "Return only the JSON object."
    )

    try:
        part = genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    except Exception:
        return {"__error__": "Could not prepare the file for reading. Try a different image."}

    raw = _gemini_generate([prompt, part], response_json=True)
    if isinstance(raw, str) and raw.startswith("__ERROR__"):
        return {"__error__": raw[len("__ERROR__"):]}

    
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return {"__error__": "The model did not return receipt fields. Try a clearer photo."}
        return data
    except Exception:
        return {"__error__": "Could not understand the receipt. Try a clearer, well-lit photo."}



page = st.sidebar.radio(
    "Page Navigation",
    ["Home", "Analysis", "Inflation Forecasting", "Crowdsourcing Receipts"],
)

st.sidebar.divider()
if ai_ready():
    _used, _cap = quota_status()
    st.sidebar.caption(f"AI features (Gemini): enabled · {_used}/{_cap} requests used today")
elif GEMINI_AVAILABLE:
    st.sidebar.caption("AI features (Gemini): add GEMINI_API_KEY to enable")
else:
    st.sidebar.caption("AI features (Gemini): run  pip install google-genai  to enable")



if page == "Home":
    st.title("Ayara: Your local market decoded. ")

    safe_image(LOGO_PATH, caption="Ayara Logo")

    st.header("Project Overview")
    st.write(""" This web-based platform turns everyday receipts into a credible source of economic insight, revealing the real cost of living at the neighborhood level. Users can upload photos of paper receipts, SMS confirmations, or PDF invoices, and the system processes them using optical character recognition and natural language processing to capture product names, quantities, prices, dates, and store details across varied formats and Indian languages. Each receipt is linked to a precise location through store information, metadata, or user input, without relying on intrusive GPS tracking. Prices are standardized for unit size, brand, and packaging, creating a fair basis for comparison between regions. The processed data is analyzed through advanced time-series models that detect both gradual inflation trends and sudden price spikes, producing a hyper-local inflation index that often reveals patterns hidden in national averages.

The platform presents this information in formats tailored to its different audiences. Citizens receive clear cost-of-living summaries and intuitive visual cues that make complex trends easy to understand. Journalists and NGOs can explore interactive maps, identify unusual price surges, and download datasets for deeper analysis. Policymakers have access to comprehensive dashboards with regional breakdowns, inflation comparisons, and early warnings for essential goods. By crowdsourcing data while safeguarding privacy through secure anonymization methods, the project creates a transparent and accessible view of inflation. It shifts economic awareness from official reports alone to a real-time, community-driven perspective, enabling more informed decisions in households, newsrooms, and government offices alike. """)

    st.divider()

    st.header("About Me")
    st.write("My name is Anand Varma Indukuri and I am an 18-year-old freshman at the University of Wisconsin–Madison studying Computer Science and Data Science. Growing up in India, I often witnessed the quiet ways inflation shaped daily life. Families would adjust their meals, postpone purchases, or stretch their budgets without fully realizing how much their expenses had grown. Official statistics rarely captured these everyday struggles, creating a gap between the numbers on paper and the reality people experienced. I wanted to create something that could close that gap by giving every household, regardless of income, a clear and honest picture of its cost of living. This project is also intended for those in government who make critical decisions, offering them precise and timely data from the ground. For me, it is not only a technical challenge but also a way to bring transparency, empowerment, and a stronger voice to communities.")



if page == "Analysis":
    st.title("Analysis of India's Hyperlocal Inflation")
    safe_image(INFLATION_IMG_PATH, caption="Hyperlocal Inflation")

    df = get_data()

    st.header("Data:")
    col1, col2 = st.columns([3, 1])

    products_selected = col1.multiselect("Select a product category", PRODUCT_LIST)
    sector_selected = col2.selectbox("Select sector", ["Rural", "Urban", "Rural+Urban"], index=2)

    relevant_cols = ["Month_Year"] + products_selected

    
    if products_selected:
        df_filtered = df[df["Sector"] == sector_selected].copy()
        df_products = df_filtered[relevant_cols].copy()
        df_products.set_index("Month_Year", inplace=True)
        df_products.sort_index(inplace=True)

        fig = plt.figure(figsize=(12, 6))
        for col in products_selected:
            plt.plot(df_products.index, df_products[col], label=col)
        plt.title(f'Selected Product CPI Over Time: {", ".join(products_selected)}', fontsize=14)
        plt.ylabel("CPI")
        plt.xlabel("Date")
        plt.legend()
        plt.grid()
        plt.xticks(rotation=90, fontsize=7)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Select one or more product categories above to see the CPI chart.")

    st.header("Analysis Shown Below: ")
    col_aud, col_cat, col_lang = st.columns(3)

    user_list = ["Government", "Regular User"]
    audience = col_aud.radio("Which analysis do you want?", user_list, index=0)
    analysis_category = col_cat.radio("Pick a category to analyze", PRODUCT_LIST, index=0)
   
    language = col_lang.radio("Language for the analysis", ["English", "Hindi", "Telugu"], index=0)

    simplify = st.checkbox("Make it shorter and easier to read (AI-powered)", value=False)

    ANALYSIS_TEXT = {
    'Government': {
        'Fruits': """ 
    1. Long-Term Inflationary Trend
     The CPI for fruits has consistently increased from approximately 103 in January 2013 to more than 150 in the most recent data, representing a cumulative rise of about 45 to 50 percent. This pattern indicates structural inflation rather than short-term fluctuations, caused by persistent factors such as rising input costs, higher fuel prices, increased labor wages, and inefficiencies in the supply chain. For the government, this trend signals the need for structural reforms in the fruit market. Measures could include expanding cold chain networks through the Mission for Integrated Development of Horticulture, enhancing market integration via the National Agriculture Market platform, and incentivizing private investment in modern storage and distribution facilities. These steps would reduce wastage, lower supply costs, and slow the rate of price increases, directly benefiting both producers and consumers while mitigating inflationary pressure on the broader economy.
 
    2. Seasonal Variation and Harvest Cycles
     The CPI trend for fruits displays recurring peaks and troughs that align with harvest and lean    periods. Prices tend to rise before harvest seasons when supply is scarce and drop when the market is saturated with fresh produce. For example, mango prices rise sharply in March and April before falling in June once the main crop enters the market. Apple prices in northern India increase between March and May as stored stocks diminish, followed by a decline in September and October during harvest. Citrus fruits such as oranges and sweet limes often see price increases during the monsoon when supply chains slow and availability decreases. Recognizing these cycles allows the government to act preemptively. Policy tools could include facilitating staggered planting to ensure more consistent supply throughout the year, releasing stored produce during shortage months, and enabling timely imports to stabilize domestic prices without harming local producers.
 
    3. Price Shocks and Market Disruptions
     Beyond predictable seasonal patterns, the CPI data shows sudden and significant price changes that point to market shocks. These can be caused by extreme weather, pest outbreaks, transportation strikes, or emergency policy decisions. During the COVID-19 lockdown in April 2020, for instance, fruit CPI in cities rose steeply due to restricted interstate movement, while rural areas experienced a glut and falling prices due to unsold produce. In 2015, unseasonal rainfall in Maharashtra damaged banana and grape crops, leading to rapid price escalations. To manage such situations, the government can establish a national real-time price and supply monitoring system integrated with meteorological data. This system could trigger rapid interventions such as emergency transportation subsidies, temporary reductions in import tariffs, and targeted distribution of essential fruits through the Public Distribution System to keep consumer prices stable and prevent farmer losses.
 
4. Urban and Rural Price Divergence
 While the aggregated CPI indicates general trends, the underlying data reveals significant price differences between rural and urban markets. Urban areas tend to have more stable fruit prices due to better infrastructure, cold storage facilities, and more reliable transport links. Rural areas, in contrast, experience sharper fluctuations, sometimes with sudden surges during shortages and steep declines during gluts. In rural Bihar in 2019, banana prices collapsed by over 40 percent in a matter of weeks due to excess supply and inadequate storage, while urban consumers in Patna saw no such reduction. Addressing these disparities requires government investment in rural cold storage hubs, improvements in rural transport networks through the Pradhan Mantri Gram Sadak Yojana, and strengthening farmer-producer organizations to connect rural supply directly to urban demand. These actions would stabilize rural prices and enhance farmer incomes while keeping urban markets supplied at consistent rates.
 
5. Contribution to Headline Inflation and Monetary Policy
 Fruits contribute significantly to the Food and Beverages category in the CPI basket, which has a strong influence on headline inflation. Persistent increases in fruit prices can raise overall CPI, prompting the Reserve Bank of India to adjust monetary policy by raising interest rates. This can slow economic growth and increase borrowing costs across sectors. In July 2023, retail inflation reached 7.44 percent, with perishables such as tomatoes, onions, and certain fruits playing a key role in the rise. For the government, monitoring fruit CPI can serve as an early warning for inflationary trends. Possible interventions include temporary easing of import restrictions for specific fruits from countries such as Thailand and Vietnam, subsidizing the transport of surplus produce from high-supply to high-demand regions, and reducing export incentives when domestic prices threaten to push overall inflation beyond acceptable limits.
 
6. Climate Change and Risk Forecasting
 The increasing intensity of CPI fluctuations in recent years suggests a growing influence of climate-related disruptions. Extreme rainfall, prolonged heatwaves, and hailstorms have all affected fruit yields, leading to sharp price changes. In 2023, apple production in Himachal Pradesh fell significantly due to reduced chill hours and unseasonal rains, causing prices in Delhi and Chandigarh to rise by over 20 percent within two months. To counter this risk, the government should integrate CPI data with weather forecasting and satellite-based crop monitoring to identify potential shortages well in advance. Investments in climate-resilient horticulture, micro-irrigation systems, and the promotion of drought- and heat-tolerant fruit varieties would strengthen supply resilience. Additionally, introducing index-based crop insurance schemes that provide payouts when fruit CPI exceeds a set threshold could protect both farmers and consumers from severe climate-driven price shocks. """,
        'Vegetables': """ Long-Term Inflationary Trend
 The CPI for vegetables has risen markedly from about 102 in January 2013 to roughly 160 in the latest period of your dataset (Rural+Urban, February 2021), a cumulative increase of approximately 55 to 60 percent. Urban markets show an even steeper climb, from about 103 to around 180 over the same span, while rural markets rise from about 102 to nearly 150. This sustained, multi-year increase indicates structural pressures in horticultural supply chains, including post-harvest losses, elevated logistics costs, and vulnerability to fuel and wage increases. For the government, the implication is to prioritize structural interventions in the vegetable value chain: accelerate cold-chain and pack-house capacity under the Mission for Integrated Development of Horticulture, expand market integration via e-NAM with time-bound onboarding of APMCs, and incentivize private investment in reefer transport and last-mile storage. These steps reduce spoilage, lower intermediation costs, and dampen price pass-throughs, moderating headline inflation without suppressing farmgate incomes.


Seasonal Variation and Monsoon Sensitivity
 The CPI for vegetables exhibits pronounced seasonal amplitude, typically softening in late winter–early summer and rising through the monsoon and post-monsoon months. In your series, average levels are lower around March–May and higher from July through December, peaking most often in October–November. This pattern reflects harvest cycles for leafy and fruit vegetables and the heightened logistics frictions during monsoon months, when weather disruptions and road conditions constrain inflows to urban mandis. A policy response should blend supply smoothing and information flow: promote staggered sowing calendars through Krishi Vigyan Kendras for key staples like tomatoes, onions, and potatoes; operationalize district-level buffer norms to release stocks during lean weeks; and deploy real-time mandi arrival dashboards to pre-emptively redirect surplus from low-price zones to tightening markets.


Price Shocks and Idiosyncratic Spikes
 Beyond seasonality, the dataset captures sharp year-on-year surges, notably in late 2019 and early 2020 when vegetable CPI spiked by more than 30 to 45 percent. These bursts align with onion-led shocks following late and excessive rains and localized flooding in major producing belts, as well as transient logistics disruptions near the onset of the pandemic. Such shocks are amplified in metros, where dependence on long-haul supply lines is higher. To manage these episodes, establish an integrated early-warning system that fuses IMD rainfall alerts, remote-sensing crop progress, and mandi inflow anomalies. This should trigger swift measures: temporary import duty adjustments or quota releases for onions and tomatoes, time-bound freight subsidies to clear chokepoints, and calibrated Strategic Buffer releases through NCCF/NAFED to stabilize retail prices while averting distress sales in producing districts.


Urban–Rural Divergence and Volatility
 Your data show a persistent urban premium over rural vegetable prices—on average around 10 index points—with occasional gaps exceeding 30 points during stress periods. Urban volatility is also higher, reflecting longer supply chains and faster demand transmission. Meanwhile, rural markets can face sudden gluts and collapses when storage and aggregation are inadequate. Policy should narrow this gap by scaling rural aggregation (FPOs/SHGs) linked to assured urban off-take, financing decentralized cold rooms at block level, and improving rural road connectivity under PMGSY to reduce transit time variability. Standardizing transparent grading at source and widening direct-to-retail channels in cities will reduce multiple handling and compress urban markups.


Contribution to Headline Inflation and Policy Coordination
 Vegetables carry meaningful weight within the Food and Beverages basket and are a frequent driver of short-run inflation volatility. The late-2019 to early-2020 surges in your series coincide with periods when food inflation pushed headline CPI higher, complicating monetary policy. For fiscal and administrative instruments to complement monetary stance, the government can operationalize dynamic buffer norms for TOP (tomato-onion-potato) and a handful of highly elastic vegetables; enable rapid inter-state movement through green corridors during stress; and adopt sunset-bound export curbs only when retail inflation risks breaching tolerance bands. This coordinated toolkit helps contain second-round effects from food spikes into non-food inflation, preserving growth while anchoring expectations.


Climate Change and Risk Forecasting
 The increasing amplitude and frequency of vegetable CPI spikes in recent years within your data suggest rising climate sensitivity—excess rainfall, heat waves, and unseasonal storms disrupt yields and storability, especially for perishables. A forward-looking approach should integrate crop-stage weather risk with market signals: scale micro-irrigation and protected cultivation for high-volatility vegetables, promote climate-resilient varieties through targeted breeder seed expansion, and link index-based insurance to market outcomes by adding CPI-or mandi-price triggers for specified districts. Coupled with satellite-based acreage nowcasts and truck-GPS feeds for corridor monitoring, this would allow pre-positioning of buffers and targeted logistics support weeks before shortages manifest in retail prices. """,
        'Spices': """ Long-Term Inflationary Trend
The CPI for spices has shown a steady upward trajectory from around 105 in January 2013 to approximately 165 in the latest data, representing a cumulative increase of about 55 to 60 percent. This long-term rise reflects structural factors, including persistent supply-demand imbalances, increased export demand from international markets, higher cultivation costs, and periodic shortages of key commodities like coriander, cumin, and turmeric. For the government, this indicates the need for supply-side reforms in the spice sector: expanding spice park infrastructure under the Spices Board, enhancing farmer access to precision irrigation and mechanized post-harvest processing, and strengthening linkages to domestic wholesale markets to reduce excessive dependence on a few trading hubs. These measures would reduce storage losses, stabilize prices, and increase domestic availability without curtailing export potential.


Seasonal Variation and Harvest Patterns
 The CPI trend for spices exhibits mild but consistent seasonal variation, with slight softening during peak harvest months and firming during lean storage periods. For example, cumin and coriander prices tend to drop between March and May following the rabi harvest, while turmeric prices often firm up from July to September due to reduced arrivals in the monsoon season. Recognizing these cycles allows the government to intervene through calibrated stock releases from accredited spice warehouses, promotion of off-season cultivation where agronomically feasible, and targeted import facilitation for high-demand spices during lean months to prevent price spikes in domestic retail markets.


Price Shocks and External Disruptions
 Beyond predictable cycles, the dataset shows sudden spikes—such as the sharp rise in late 2016 and mid-2020—driven by factors like unseasonal rains damaging standing crops, pest infestations, or surges in export orders. During the COVID-19 period, logistical bottlenecks and container shortages drove up spice prices in both domestic and export markets. The government could mitigate such volatility through a spice-specific early-warning and response framework, combining satellite crop health monitoring with export order tracking, and enabling rapid policy switches—like adjusting minimum export prices or extending freight subsidies to maintain domestic availability.


Urban–Rural Price Divergence
 Your CPI data show that urban retail prices for spices are consistently 5–15 index points higher than rural prices, due largely to branding, packaging costs, and more fragmented retail chains in cities. In rural markets, prices tend to be lower but more volatile due to dependence on weekly haats and limited storage. Reducing these disparities requires government facilitation of direct procurement channels between rural producer groups and urban supermarkets, investment in spice grinding and packaging units at district level, and the adoption of standardized grading norms to reduce variability in quality and price.


Contribution to Headline Inflation and Export Sensitivity
 While spices have a smaller weight in the CPI basket compared to staples like cereals or vegetables, their price surges can still contribute to food inflation, particularly when combined with simultaneous increases in other perishables. The export sensitivity of spices makes domestic prices vulnerable to international demand swings, as seen during periods when Indian turmeric and cumin shipments surged to the Middle East and Europe. For macroeconomic stability, the government should maintain a flexible export policy with built-in domestic availability triggers, expand buffer holdings for select spices with volatile prices, and integrate spice market intelligence into the Reserve Bank’s inflation monitoring to anticipate spillover effects.


Climate Change and Production Risk
 Increasing variability in rainfall patterns, higher average temperatures, and greater pest pressure are emerging risks for spice production. In recent years, erratic monsoon distribution has delayed sowing in Gujarat’s cumin belt and reduced turmeric yields in Telangana. The government should integrate spice crop CPI trends with IMD’s seasonal forecasts to anticipate shortages, promote heat- and drought-tolerant spice varieties through public breeding programs, and expand coverage of weather-based crop insurance tailored for high-value spices. Additionally, investments in solar-powered dehydration and storage technology can reduce post-harvest losses during prolonged wet periods, ensuring that price spikes from climate shocks are minimized.
 """,
        'Fuel and light': """ Long-Term Inflationary Trend
 The CPI for Fuel and Light has risen from roughly 102 in January 2013 to about 160 in the latest available data, representing a cumulative increase of nearly 55 to 60 percent. This steady climb is largely structural, driven by global crude oil price trends, the depreciation of the rupee, periodic revisions in domestic fuel taxes, and gradual increases in electricity tariffs. The impact of regulated pricing means that CPI movements in this category often reflect policy decisions as much as market forces. For the government, this underscores the importance of diversifying the national energy mix—accelerating investments in renewable energy under the National Solar Mission, expanding LNG import and distribution capacity to cushion crude oil shocks, and implementing targeted subsidies for low-income households that ensure affordability without distorting market efficiency.


Seasonal Variation and Consumption Patterns
 While less seasonal than food categories, Fuel and Light shows minor cyclical changes linked to heating demand in winter months, festive season electricity usage, and pre-monsoon power surges. Rural areas, particularly in northern states, often record higher solid fuel prices in winter due to increased demand for firewood and biomass. Recognizing these patterns, the government could coordinate seasonal stockpiling of LPG cylinders in high-demand districts, incentivize efficient biomass stoves in rural households, and encourage time-of-day electricity pricing to manage peak load periods without triggering sharp CPI increases.


Price Shocks and External Disruptions
 The dataset reflects sudden spikes, such as in mid-2018 and late-2021, coinciding with global crude oil surges and revisions in excise duties. Other notable upticks align with currency depreciation episodes, which raise import costs for crude and LPG. In extreme cases, such as during early 2022 geopolitical tensions, wholesale and retail fuel prices in India rose sharply despite government excise cuts, highlighting supply chain and import dependency risks. The government can mitigate such shocks through a strategic petroleum reserve expansion, dynamic excise duty adjustment mechanisms tied to global benchmarks, and forward-contracting LNG imports to lock in stable prices.


Urban–Rural Divergence and Infrastructure Gaps
 Rural CPI for Fuel and Light often exceeds urban levels for kerosene, firewood, and other solid fuels due to limited supply points and higher transport costs, while urban consumers pay more for electricity and piped gas services. The dataset shows this divergence widening during logistics disruptions, such as monsoon-related road inaccessibility in rural areas. To close this gap, the government should expand last-mile LPG delivery infrastructure under the Pradhan Mantri Ujjwala Yojana, strengthen rural electricity distribution via Deen Dayal Upadhyaya Gram Jyoti Yojana, and promote decentralized renewable systems (solar micro-grids, biogas plants) to reduce dependency on transported fuels.


Contribution to Headline Inflation and Policy Signalling
 Fuel and Light carries significant weight in the CPI basket, meaning persistent increases can directly push headline inflation higher. For example, in mid-2021, rising petrol, diesel, and LPG prices contributed materially to headline CPI breaching the RBI’s 6 percent upper tolerance band, forcing monetary policy recalibration. Since these costs also feed into transport and production, they have indirect inflationary effects across the economy. For policy stability, the government could adopt a transparent fuel pricing formula that smooths volatility through variable taxes, coordinate fuel price interventions with monetary authorities, and release advance tariff change schedules for electricity to allow businesses and households to adjust.


Climate Change, Transition Risk, and Energy Security
 As India transitions towards a low-carbon economy, managing CPI stability during the shift is critical. The rollout of renewables, while reducing long-term fossil fuel exposure, must be balanced with short-term affordability. Extreme weather events—such as extended heatwaves—increase electricity demand and stress supply, driving tariff hikes. Meanwhile, global climate policies can influence fossil fuel availability and cost. To manage these transition risks, the government should integrate CPI trends into the National Energy Policy’s forecasting, accelerate grid-scale battery storage to stabilize renewable output, and promote electrification of cooking and transport to reduce LPG and petrol dependency. Such measures would enhance both price stability and energy resilience.
 """,
        'Health': """ Long-Term Inflationary Trend
 The CPI for Health has shown a steady rise from about 102 in January 2013 to nearly 165 in the latest data, representing a cumulative increase of roughly 60 percent over the period. Unlike food and fuel, this trend is less influenced by short-term market volatility and more by structural cost drivers: rising prices of pharmaceuticals, higher consultation and diagnostic fees, increased cost of imported medical equipment, and wage growth in the healthcare workforce. For the government, this underscores the need for systemic reforms to moderate healthcare inflation—strengthening the Pradhan Mantri Jan Arogya Yojana (PM-JAY) to expand free coverage, promoting domestic manufacturing of medical devices under the Production Linked Incentive (PLI) scheme, and incentivizing generic drug adoption through Jan Aushadhi Kendras to lower retail medicine prices.


Seasonal Variation and Epidemic Cycles
 Health CPI shows mild seasonal patterns tied to disease outbreaks and climate-linked health events. Monsoon months often see increased demand for diagnostic services and medicines due to vector-borne diseases like dengue and malaria, leading to temporary price rises in pathology and treatment services. Winter months can see a similar uptick in respiratory illness-related spending. Policy measures should include advance procurement of essential medicines ahead of high-incidence seasons, mobile diagnostic units in high-risk districts, and fast-track insurance claim processing during epidemic months to avoid cost escalation from delayed care.


Price Shocks and External Disruptions
 The dataset captures noticeable upward jumps, such as during the COVID-19 pandemic, when the costs of PPE, oxygen supply, hospital stays, and diagnostics surged sharply. Similar localized shocks have been observed when imported raw materials for drugs (Active Pharmaceutical Ingredients from China) faced supply chain disruptions. The government can pre-empt such situations by diversifying API sourcing, building national reserves of critical medical consumables, and mandating transparent hospital pricing guidelines during declared public health emergencies to prevent opportunistic overcharging.


Urban–Rural Disparities and Accessibility Gaps
 Urban areas often exhibit higher CPI levels for healthcare due to greater reliance on private providers, higher real estate costs for hospitals, and premium pricing for branded drugs. Rural CPI for health is typically lower but masks hidden costs such as travel to distant facilities and delays in treatment. Addressing these gaps requires government investment in rural health infrastructure under the Ayushman Bharat Health and Wellness Centres initiative, telemedicine expansion to bridge specialist shortages, and subsidized transportation for patients requiring referral care in district hospitals.


Contribution to Headline Inflation and Social Stability
 Although the Health category carries a moderate weight in the CPI basket, its persistent inflation can have disproportionate social impact, as rising healthcare costs directly reduce disposable income and can push vulnerable households into debt. During periods of simultaneous food and healthcare inflation, the burden on low-income groups is particularly acute. For macroeconomic stability, the government could implement health cost monitoring alongside food inflation tracking, introduce price caps on essential medical services during crises, and expand preventive care programs that reduce high-cost interventions in the long run.


Climate Change and Emerging Health Risks
 The intersection of climate change and public health is increasingly visible in CPI movements, as climate-related disease burdens raise demand for medical services and pharmaceuticals. Heatwaves drive demand for hydration therapy and heat illness treatment; flooding increases vector-borne diseases; and pollution spikes increase respiratory care costs. To manage this, the government should integrate CPI health trends with climate and epidemiological forecasting, invest in public health resilience infrastructure (flood-proof hospitals, climate-controlled wards), and promote community-level preventive health programs. Additionally, supporting domestic vaccine R&D and manufacturing for climate-sensitive diseases could reduce import dependence and future price shocks. """,
    },
    'Regular User': {
        'Fruits': """ 1. Fruit Prices Have Been Rising for Years:
 Fruit prices have gone up by nearly 50 percent since 2013. This means buying the same amount of fruits today costs much more than it did ten years ago. For families, this is a reminder to look for cheaper seasonal fruits and buy in bulk when prices are lower. Storing fruits like apples or oranges in a fridge for longer use can help stretch your budget.
 
2. Prices Change With the Seasons:
 Certain fruits are cheap when they are in season and costlier when they are not. Mangoes, for example, are much cheaper in May and June than in March, while apples are most affordable around September and October. Knowing these cycles helps families plan their fruit shopping for the right months, saving money while enjoying fresher produce.
 
3. Sudden Price Spikes Can Happen:
 Sometimes fruit prices jump unexpectedly due to bad weather, transport problems, or reduced supply. This happened during the COVID-19 lockdown when many fruits became very expensive in cities. If prices suddenly rise, consider switching to other fruits or vegetables that are cheaper at the time until prices come down again.
 
4. City and Village Prices Can Be Different:
 Fruits can be much cheaper in rural areas where they are grown, while in cities prices are often higher due to transport and storage costs. If possible, buy directly from farmers' markets or weekly haats instead of supermarkets. This not only saves money but also supports local growers.
 
5. Fruits Affect the Cost of Living:
 When fruit prices go up, it can push up the cost of other things too, because fruits are part of the official cost-of-living measure. If you notice fruit prices climbing for several months, it’s a sign other daily essentials may also get costlier. Households can prepare by adjusting monthly budgets earlier rather than being caught off guard.
 
6. Weather and Climate Play a Big Role:
 Fruits depend heavily on the weather. Heatwaves, floods, or heavy rains can reduce supply and increase prices. If the news predicts poor weather for fruit-growing regions, expect prices to rise soon. Families can buy and store more durable fruits in advance or switch to alternatives that are less affected by weather problems. """,
        'Vegetables': """ Vegetable Prices Have Increased Over Time
 Since 2013, vegetable prices have gone up by around 55–60 percent. This means your monthly vegetable spending is higher now than a decade ago. Buying vegetables in local markets or directly from farmers can often be cheaper than supermarkets.


Prices Change With the Seasons
 Vegetables are cheapest during harvest time and more expensive in off-season months. Tomatoes, onions, and potatoes, for example, are cheaper just after harvest. Knowing these cycles can help families buy in bulk and store long-lasting vegetables like potatoes and onions for later use.


Sudden Spikes Can Happen
 Prices can rise sharply due to heavy rains, crop diseases, or transport strikes. When prices jump, try switching to alternative vegetables that are in season and cheaper at the time.


City and Village Prices Can Differ
 In villages, vegetables are often cheaper when they are grown locally, while in cities prices are higher due to storage and transport costs. Visiting weekly markets or buying directly from farmers can save money.


Vegetables Affect the Cost of Living
 Since vegetables are part of the cost-of-living index, a rise in vegetable prices can signal that other foods might get more expensive soon. If prices are rising steadily, plan your monthly budget in advance.


Weather Plays a Big Role
 Vegetable supply depends heavily on the weather. Floods, droughts, or unseasonal rains can lead to shortages and higher prices. If bad weather is expected, stock up on durable vegetables or frozen options before prices rise. """,
        'Spices': """ Spice Prices Have Been Rising for Years
 Since 2013, the cost of spices like turmeric, cumin, and coriander has increased by over 55 percent. This means everyday cooking is more expensive now than a decade ago. Buying in bulk during harvest season and storing in airtight containers can help save money.


Prices Change With the Harvest
 Spices are cheaper after harvest—for example, cumin and coriander prices drop between March and May—while prices rise in months with lower supply. Buying your annual stock soon after harvest can help you get better prices.


Unexpected Price Jumps Can Happen
 Poor weather, pests, or increased export demand can cause spice prices to rise suddenly. If this happens, try substituting with other spices or reducing usage in recipes until prices settle.


City and Village Price Gaps
 Urban markets often sell branded spices at higher prices, while in rural areas loose spices can be cheaper but may vary in quality. Families can save by buying from trusted wholesale markets and storing at home.


Spices Affect the Cost of Living
 Even though spices are a smaller part of the household budget, sharp increases can raise the overall cost of meals, especially when combined with rising vegetable or oil prices. If spice prices are going up, plan ahead for grocery costs.


Weather Impacts Spice Supply
 Heavy rains, drought, or heat can damage spice crops, leading to higher prices. If you hear about poor weather in spice-growing states, buy and store the spices you use most before prices rise.
 """,
        'Fuel and light': """ Fuel Prices Have Been Rising for Years
 Since 2013, the cost of fuel and electricity has gone up by more than 50 percent. This means filling your vehicle, using LPG gas, or paying electricity bills is more expensive now than a decade ago. Families can save by reducing unnecessary trips, using energy-efficient appliances, and cooking with pressure cookers to save gas.


Small Seasonal Changes Can Happen
 Fuel and electricity prices may go up slightly in certain months—like during winter in cold areas or during festivals when electricity use is high. Planning ahead by reducing usage in high-demand months can help keep bills under control.


Sudden Price Jumps Can Occur
 Global oil prices, currency changes, or government tax changes can quickly make fuel more expensive. When this happens, consider using public transport more often, carpooling, or combining errands into one trip to cut costs.


City and Village Price Differences
 In rural areas, firewood and kerosene can be more expensive due to transport costs, while in cities electricity and piped gas tend to cost more. Families should choose the most affordable option available in their area and avoid wasting fuel.


Fuel Costs Affect Everything Else
 When fuel prices rise, the cost of transporting goods also increases, which can make groceries and other daily needs more expensive. If fuel prices stay high for several months, expect other items to get costlier too, and adjust your budget accordingly.


Weather and Energy Use
 Heatwaves, cold spells, and heavy rains can all increase fuel and electricity use. Using solar water heaters, LED lights, and energy-saving fans can reduce bills during extreme weather and make households less dependent on expensive energy sources. """,
        'Health': """ Healthcare Costs Keep Climbing
 Since 2013, the cost of medical care has gone up by nearly 60 percent. This means doctor visits, medicines, and hospital treatments are much more expensive now than ten years ago. Families can save by using government health schemes like Ayushman Bharat or by buying medicines from Jan Aushadhi stores, which sell affordable generic drugs.


Seasonal Illnesses Can Affect Your Budget
 Medical expenses often rise during certain times of the year—like the monsoon, when diseases such as dengue and malaria become more common, or in winter, when respiratory illnesses increase. Families can prepare by keeping essential medicines and basic health supplies at home before these seasons start.


Unexpected Medical Price Jumps
 During emergencies, like the COVID-19 pandemic, the cost of medicines, oxygen, and hospital stays can rise sharply. Keeping basic first-aid items and common medicines at home can help avoid high prices during crises, and having health insurance can protect against sudden large bills.


Costs Vary Between Cities and Villages
 In cities, healthcare is often more expensive because people rely on private hospitals, while in villages costs are lower but facilities may be far away. If possible, families should use government hospitals for non-urgent care or preventive check-ups to reduce costs.


Health Costs Affect the Cost of Living
 As medical prices go up, they make life more expensive overall. If you notice medical bills getting higher month after month, adjust your family budget to set aside more for health expenses, even if you are not sick at the moment.


Weather and Climate Can Impact Health Spending
 Extreme heat, heavy rains, or floods can lead to more illnesses, which can raise medical costs. If bad weather is predicted in your area, it’s smart to take preventive health steps—like boiling water, using mosquito nets, or wearing masks—to reduce the risk of falling sick and spending more on treatment. """,
    }
}

    
    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False
    if "analysis_audience" not in st.session_state:
        st.session_state.analysis_audience = None
    if "analysis_categories" not in st.session_state:
        st.session_state.analysis_categories = []

    btn1_col, btn2_col = st.columns([1, 1])
    show_one = btn1_col.button("Generate analysis for chosen audience & category")
    show_all = btn2_col.button("Generate analysis for ALL 5 categories (chosen audience)")

    if show_one:
        st.session_state.show_analysis = True
        st.session_state.analysis_audience = audience
        st.session_state.analysis_categories = [analysis_category]

    if show_all:
        st.session_state.show_analysis = True
        st.session_state.analysis_audience = audience
        st.session_state.analysis_categories = PRODUCT_LIST[:]

    
    if not st.session_state.show_analysis:
        st.info("Choose an audience and category above, then click a Generate button to see the analysis.")
    else:
        chosen_audience = st.session_state.analysis_audience
        chosen_categories = st.session_state.analysis_categories

        if not chosen_categories:
            st.info("No categories selected. Click one of the buttons above to generate analysis.")
        else:
            if len(chosen_categories) < len(PRODUCT_LIST):
                subtitle = ", ".join(chosen_categories)
            else:
                subtitle = "All 5 categories"
            lang_note = f" — in {language}" if language != "English" else ""
            st.subheader(f"Analysis for: {chosen_audience} — {subtitle}{lang_note}")

            for cat in chosen_categories:
                with st.expander(f"{chosen_audience} • {cat}", expanded=True):
                    original = ANALYSIS_TEXT.get(chosen_audience, {}).get(cat, "No analysis available.")
                    need_ai = (language != "English") or simplify

                    if not need_ai:
                        st.markdown(original)
                    else:
                        with st.spinner(f"Preparing the {cat} analysis in {language}..."):
                            result = ai_translate_simplify(original, language, simplify)

                        if result is None:
                            st.warning(
                                "AI translation/simplification needs the Gemini SDK and an API key, "
                                "so the original English text is shown below. See the setup notes at "
                                "the top of the script to enable it (pip install google-genai and add "
                                "GEMINI_API_KEY)."
                            )
                            st.markdown(original)
                        elif isinstance(result, str) and result.startswith("__ERROR__"):
                            st.error(
                                f"The AI request failed ({result[len('__ERROR__'):]}). "
                                "Showing the original English text instead."
                            )
                            st.markdown(original)
                        else:
                            st.markdown(result)
                        with st.expander("Show original (English)"):
                                st.markdown(original)



if page == "Inflation Forecasting":
    st.title("Forecasting Product CPI")
    col1, col2, col3, col4 = st.columns(4)

    forecast_type = col1.selectbox("How you want to see forecast", ["Specific date", "Over time"])
    product = col2.selectbox("Select product category", PRODUCT_LIST)
    sector = col3.selectbox("Select a sector", SECTORS)
    dt = col4.date_input("Forecast for date", format="YYYY-MM-DD")

    
    if dt is None:
        st.info("Please choose a date to forecast for.")
        st.stop()

    selected_month_start = pd.to_datetime(dt).replace(day=1)

    st.subheader(
        f'Forecasting for {sector} {product} CPI for the month of '
        f'{selected_month_start.strftime("%B %Y")}'
    )

    
    df = get_data()
    sector_df = df[df["Sector"] == sector].copy()
    sector_df.set_index("Month_Year", inplace=True)
    
    sector_df.index = sector_df.index.to_period("M").to_timestamp()
    sector_df.sort_index(inplace=True)
    
    sector_df = sector_df[~sector_df.index.duplicated(keep="first")]

    if product not in sector_df.columns:
        st.error(f"'{product}' is not a column in the dataset, so it cannot be looked up or forecast.")
        st.stop()

    
    data_start = sector_df.index.min()
    data_end = sector_df.index.max()

    if selected_month_start < data_start:
        
        st.warning(
            f'{selected_month_start.strftime("%B %Y")} is before the data begins '
            f'({data_start.strftime("%B %Y")}). Forecasting only works for dates after the '
            "available data, so no value can be shown for this month."
        )

    elif selected_month_start <= data_end:
        
        if selected_month_start in sector_df.index:
            value = sector_df.at[selected_month_start, product]
            if pd.isna(value):
                st.info(
                    f'{selected_month_start.strftime("%B %Y")} is within the data range, but there is '
                    f"no recorded {product} value for that month."
                )
            else:
                st.success(f'{selected_month_start.strftime("%B %Y")} is within the data range.')
                st.write(
                    f"The **{sector.lower()} {product.lower()} CPI** for the month of "
                    f'**{selected_month_start.strftime("%B %Y")}** is **{value:.2f}**.'
                )
        else:
            st.info(
                f'{selected_month_start.strftime("%B %Y")} falls inside the overall date range but '
                "is not present in the dataset (missing month), so no value can be shown."
            )

    else:
        
        st.warning(
            f'{selected_month_start.strftime("%B %Y")} is beyond the available data, so it will be forecast.'
        )

        month_diff = (selected_month_start.year - data_end.year) * 12 + (
            selected_month_start.month - data_end.month
        )

        if month_diff <= 0:
            st.error("Could not determine a valid forecast horizon for this date.")
        elif (product, sector) not in MODEL_PATHS:
            st.error(f"No forecasting model is available yet for {product} ({sector}).")
        else:
            model_fp = MODEL_PATHS[(product, sector)]
            try:
                model = joblib.load(model_fp)
                forecasts = model.get_forecast(steps=month_diff)

                if forecast_type == "Specific date":
                    predicted = forecasts.predicted_mean.iloc[-1]
                    st.write(
                        f"The forecasted **{sector.lower()} {product.lower()} CPI** for the month of "
                        f'**{selected_month_start.strftime("%B %Y")}** is **{predicted:.2f}**.'
                    )
                else:
                    future_mean = forecasts.predicted_mean
                    future_ci = forecasts.conf_int()

                    fig = plt.figure(figsize=(10, 5))
                    plt.plot(sector_df.index, sector_df[product], label="Observed")
                    plt.plot(future_mean.index, future_mean, label="Forecast", color="green")
                    plt.fill_between(
                        future_ci.index,
                        future_ci.iloc[:, 0],
                        future_ci.iloc[:, 1],
                        color="green",
                        alpha=0.2,
                    )
                    plt.title(f"{sector} {product} CPI Forecast")
                    plt.xlabel("Date")
                    plt.ylabel("CPI")
                    plt.grid(True)
                    plt.legend(loc="upper left")
                    st.pyplot(fig)
                    plt.close(fig)

            except FileNotFoundError:
                st.error(
                    f"The model file could not be found at:\n{model_fp}\n\n"
                    "Check MODELS_DIR and the file name in the CONFIG block."
                )
            except Exception as e:
                st.error(f"Model not ready yet: {e}")



if page == "Crowdsourcing Receipts":
    st.title("Crowdsource Your Receipts")
    st.write(
        "Help build a real-time, community-driven picture of local prices by sharing your receipts. "
        "Upload a clear photo of a paper receipt (PNG/JPG) or a PDF invoice, and the app uses AI to "
        "read the merchant, date, total, and line items automatically."
    )

    if not ai_ready():
        if not GEMINI_AVAILABLE:
            st.warning(
                "Automatic reading needs the Gemini SDK. Install it with "
                "`pip install google-genai` and restart the app. You can still upload files below."
            )
        else:
            st.warning(
                "Automatic reading needs a Gemini API key. Add GEMINI_API_KEY to "
                "`.streamlit/secrets.toml` (see the setup note at the top of the script). "
                "You can still upload files below; they just won't be read automatically yet."
            )

    _used, _cap = quota_status()
    st.caption(f"Free-tier usage today: {_used}/{_cap} AI requests (auto-limited to protect your quota).")

    uploaded = st.file_uploader(
        "Upload receipt(s) — photo, screenshot, or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("No receipts uploaded yet. Choose one or more files above to contribute.")
    else:
        if len(uploaded) > MAX_FILES_PER_RUN:
            st.warning(
                f"You uploaded {len(uploaded)} files; only the first {MAX_FILES_PER_RUN} "
                "will be processed at once to stay inside the free tier."
            )

        all_items = []
        for f in uploaded[:MAX_FILES_PER_RUN]:
            st.divider()
            st.subheader(f"Receipt: {f.name}")

            raw_bytes = f.getvalue()
            size_mb = len(raw_bytes) / (1024 * 1024)
            if size_mb > MAX_UPLOAD_MB:
                st.error(
                    f"This file is {size_mb:.1f} MB, over the {MAX_UPLOAD_MB} MB limit. "
                    "Please upload a smaller or more compressed file."
                )
                continue

            is_pdf = f.name.lower().endswith(".pdf") or (f.type == "application/pdf")

            
            if is_pdf:
                send_bytes, mime = raw_bytes, "application/pdf"
                st.caption("PDF detected — sending to the reader.")
            elif PIL_AVAILABLE:
                try:
                    img = Image.open(io.BytesIO(raw_bytes))
                    st.image(img, caption=f.name, width=320)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90)
                    send_bytes, mime = buf.getvalue(), "image/jpeg"
                except Exception:
                    st.error("That image could not be opened. Please try a different file.")
                    continue
            else:
                send_bytes, mime = raw_bytes, (f.type or "image/jpeg")

            if not ai_ready():
                st.info("File received. Add a Gemini API key to read it automatically.")
                continue

            with st.spinner("Reading the receipt with AI..."):
                data = ocr_receipt(send_bytes, mime)

            if "__error__" in data:
                st.error(data["__error__"])
                continue

            
            c1, c2, c3 = st.columns(3)
            c1.metric("Merchant", str(data.get("merchant_name") or "—"))
            c2.metric("Date", str(data.get("transaction_date") or "—"))
            total = data.get("total")
            currency = data.get("currency") or ""
            c3.metric("Total", f"{currency} {total}".strip() if total is not None else "—")

            
            items = data.get("items") or []
            if items:
                try:
                    items_df = pd.DataFrame(items)
                    items_df.insert(0, "merchant", data.get("merchant_name"))
                    items_df.insert(1, "date", data.get("transaction_date"))
                    items_df["currency"] = data.get("currency")
                    st.dataframe(items_df, use_container_width=True)
                    all_items.append(items_df)
                except Exception:
                    st.json(data)
            else:
                st.caption("No line items were detected on this receipt.")
                st.json(data)

        
        if all_items:
            combined = pd.concat(all_items, ignore_index=True)
            st.divider()
            st.subheader("All parsed items")
            st.dataframe(combined, use_container_width=True)
            st.download_button(
                "Download parsed items as CSV",
                data=combined.to_csv(index=False).encode("utf-8"),
                file_name="ayara_parsed_receipts.csv",
                mime="text/csv",
            )
            st.caption(
                "These results are kept only for this session. "
                + (
                    "Optional Google Sheets logging can be added (gspread is installed)."
                    if GSPREAD_AVAILABLE
                    else "To log them to the cloud, install gspread and add service-account credentials."
                )
            )
