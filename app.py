import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# --- 1. โหลดโมเดล (ตัวเดียว) ---
@st.cache_resource
def load_models():
    print("...กำลังโหลดโมเดล (จะทำแค่ครั้งเดียว)...")
    try:
        # --- แก้ไข ---
        model = joblib.load("multi_pig_model_v3.pkl")
        return model, True
    except FileNotFoundError:
        st.error("ไม่พบไฟล์โมเดล! กรุณาตรวจสอบว่า `multi_pig_model_v3.pkl` อยู่ในโฟลเดอร์เดียวกับ `app.py`")
        return None, False

# --- แก้ไข ---
multi_model_v3, models_loaded = load_models()


# --- 2. ฟังก์ชัน "ทำนาย" (V3 - โมเดลเดียว) ---
# นี่คือ "เครื่องมือ" ที่นักออกแบบ V4 (Bayesian) จะเรียกใช้
def get_pig_prediction_v3(breed, sweet_potato, oak, days, 
                           temperature, walk_distance, density, air_quality):
    
    # เตรียม Input 8 อย่าง
    input_data = {
        'percent_sweet_potato': [sweet_potato], 'percent_oak': [oak], 'feed_days': [days],
        'temperature': [temperature], 'walk_distance_km': [walk_distance],
        'stocking_density': [density], 'air_quality': [air_quality],
        'breed_Kurobuta': [1 if breed == "Kurobuta" else 0],
        'breed_Iberian': [1 if breed == "Iberian" else 0]
    }
    model_columns = [
        'percent_sweet_potato', 'percent_oak', 'feed_days',
        'temperature', 'walk_distance_km', 'stocking_density', 'air_quality',
        'breed_Kurobuta', 'breed_Iberian'
    ]
    input_df = pd.DataFrame(input_data, columns=model_columns)

    # --- แก้ไข ---
    # สั่ง AI (ตัวเดียว) ทำนาย
    prediction_array = multi_model_v3.predict(input_df)
    results = prediction_array[0]
    
    # รวบผลลัพธ์ 6 อย่างกลับไป
    output = {
        "predicted_imf": float(results[0]),
        "predicted_sweetness": float(results[1]),
        "predicted_aroma": float(results[2]),
        "predicted_umami": float(results[3]),
        "predicted_fat_quality": float(results[4]),
        "predicted_ph_24h": float(results[5]) # (แม้จะไม่แม่น แต่ก็ต้องแสดงผล)
    }
    return output

# --- 3. โค้ด "นักออกแบบ V4" (Bayesian Optimizer) ---
# (ส่วนนี้เหมือนเดิม เพราะมันเรียกใช้ get_pig_prediction_v3)

# 3.1 กำหนด "ขอบเขตการค้นหา" (Search Space)
space = [
    Categorical(['Kurobuta', 'Iberian'], name='breed'),
    Real(0.0, 60.0, name='sweet_potato'),
    Real(0.0, 40.0, name='oak'),
    Integer(120, 221, name='days'),
    Real(10.0, 35.0, name='temperature'),
    Real(0.1, 2.5, name='walk_distance'),
    Real(0.7, 1.5, name='density'),
    Real(1.0, 10.0, name='air_quality')
]

# 3.2 สร้าง "ฟังก์ชันเป้าหมาย" (Objective Function)
TARGETS = {}

@use_named_args(space)
def objective_function(**params):
    # --- แก้ไข ---
    # เรียก "เครื่องมือ" (V3 - โมเดลเดียว)
    prediction = get_pig_prediction_v3(
        params['breed'], params['sweet_potato'], params['oak'], params['days'],
        params['temperature'], params['walk_distance'], params['density'], params['air_quality']
    )
    
    # คำนวณ "ความพลาด" (Error)
    error_imf = abs(prediction['predicted_imf'] - TARGETS['imf']) * 2.0
    error_sweet = abs(prediction['predicted_sweetness'] - TARGETS['sweet'])
    error_aroma = abs(prediction['predicted_aroma'] - TARGETS['aroma'])
    error_umami = abs(prediction['predicted_umami'] - TARGETS['umami']) * 1.5
    error_fat_q = abs(prediction['predicted_fat_quality'] - TARGETS['fat_q'])
    error_ph = abs(prediction['predicted_ph_24h'] - TARGETS['ph']) * 1.5
    
    total_error = (error_imf + error_sweet + error_aroma + 
                   error_umami + error_fat_q + error_ph)
    return total_error

# 3.3 ฟังก์ชัน "นักออกแบบ" V4 (ตัวห่อหุ้ม)
def find_best_recipe_v4_bayesian(targets_dict, n_calls=100):
    global TARGETS
    TARGETS = targets_dict
    
    result = gp_minimize(
        func=objective_function, dimensions=space,
        n_calls=n_calls, random_state=42, n_jobs=-1
    )
    
    best_params_list = result.x
    best_recipe = {
        "breed": best_params_list[0],
        "percent_sweet_potato": round(best_params_list[1], 2),
        "percent_oak": round(best_params_list[2], 2),
        "feed_days": best_params_list[3],
        "temperature": round(best_params_list[4], 2),
        "walk_distance_km": round(best_params_list[5], 2),
        "stocking_density": round(best_params_list[6], 2),
        "air_quality": round(best_params_list[7], 2)
    }
    
    # --- แก้ไข ---
    # เรียก "เครื่องมือ" (V3 - โมเดลเดียว) ครั้งสุดท้าย
    best_prediction = get_pig_prediction_v3(
        best_recipe["breed"], best_recipe["percent_sweet_potato"], 
        best_recipe["percent_oak"], best_recipe["feed_days"],
        best_recipe["temperature"], best_recipe["walk_distance_km"],
        best_recipe["stocking_density"], best_recipe["air_quality"]
    )
    
    return best_recipe, best_prediction

# --- 4. สร้าง UI (หน้าเว็บ) ---
# (ส่วนนี้เหมือนเดิมเป๊ะ)
st.title("🐷 AI นักออกแบบสูตรหมู (V4: Bayesian)")

if models_loaded:
    st.sidebar.header("🎯 กรุณาป้อน 'เป้าหมาย' รสชาติ V3")
    
    st.sidebar.subheader(" (1/2) เป้าหมายรสชาติ")
    in_imf = st.sidebar.slider("IMF ที่ต้องการ (%)", 3.0, 20.0, 12.0)
    in_sweet = st.sidebar.slider("ความหวาน (1-10)", 1.0, 10.0, 8.0)
    in_aroma = st.sidebar.slider("ความหอม (1-10)", 1.0, 10.0, 9.0)
    
    st.sidebar.subheader(" (2/2) เป้าหมายคุณภาพ")
    in_umami = st.sidebar.slider("อูมามิ (1-10)", 1.0, 10.0, 8.0)
    in_fat_q = st.sidebar.slider("คุณภาพไขมัน (1-10)", 1.0, 10.0, 7.0)
    in_ph = st.sidebar.slider("pH (5.4-6.2)", 5.4, 6.2, 5.7, step=0.1)
    
    n_calls = st.sidebar.number_input("จำนวนครั้งที่ AI จะค้นหา (ความฉลาด)", 50, 500, 100)

    if st.sidebar.button("🧬 เริ่มค้นหาสูตรที่ดีที่สุด (V4)"):
        
        user_targets = {
            'imf': in_imf, 'sweet': in_sweet, 'aroma': in_aroma,
            'umami': in_umami, 'fat_q': in_fat_q, 'ph': in_ph
        }
        
        st.header("--- [ AI กำลังทำงาน ] ---")
        st.subheader("🎯 เป้าหมายของคุณคือ:")
        st.json(user_targets)

        with st.spinner(f"AI กำลัง 'ค้นหาอย่างฉลาด' {n_calls} ครั้ง... (Bayesian Optimization)"):
            start_time = time.time()
            recipe, result = find_best_recipe_v4_bayesian(user_targets, n_calls)
            end_time = time.time()

        st.success(f"ค้นหาสำเร็จ! (ใช้เวลา {end_time - start_time:.2f} วินาที)")
        st.header("--- [ ผลลัพธ์จาก AI ] ---")
        
        st.subheader("🔬 สูตรอาหารและสภาพแวดล้อมที่แนะนำ:")
        st.json(recipe)
        
        st.subheader("✅ ผลลัพธ์ที่คาดว่าจะได้:")
        st.json(result)
else:
    st.error("ไม่สามารถเริ่มแอปได้: โหลดโมเดลไม่สำเร็จ")