import gradio as gr
import numpy as np
import joblib


model = joblib.load('fashion_size_model.pkl')
le_category = joblib.load('category_encoder.pkl')
le_group = joblib.load('group_encoder.pkl')


def predict_size(target_group, category, weight, height, age):
    try:
        
        encoded_group = le_group.transform([target_group])[0]
        encoded_cat = le_category.transform([category])[0]
        
        
        features = np.array([[weight, height, age, encoded_cat, encoded_group]])
        
       
        predicted_size = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = np.max(probabilities) * 100
        
        return f" Recommended Size: {predicted_size}", f"🧠 Confidence: {confidence:.2f}%"
    except Exception as e:
        return "Error in prediction", str(e)

# 3. بناء واجهة Gradio الشيك
all_categories = list(le_category.classes_)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("#  AI Smart Size Predictor")
    gr.Markdown("### Capstone Project - Graduation Live Demo")
    
    with gr.Row():
        with gr.Column():
            target_group = gr.Dropdown(choices=["Men", "Women", "Kids"], value="Men", label="👤 Target Group")
            category = gr.Dropdown(choices=all_categories, value=all_categories[0], label="🎯 Clothing Category")
            weight = gr.Slider(minimum=15, maximum=120, value=70, step=1, label="⚖️ Weight (KG)")
            height = gr.Slider(minimum=90, maximum=200, value=170, step=1, label="📏 Height (CM)")
            age = gr.Slider(minimum=4, maximum=60, value=25, step=1, label="🎂 Age")
            submit_btn = gr.Button("⚡ Predict Optimal Size", variant="primary")
            
        with gr.Column():
            output_size = gr.Textbox(label="Recommended Size", placeholder="Result will appear here...")
            output_conf = gr.Textbox(label="Model Confidence Score")

    submit_btn.click(
        fn=predict_size,
        inputs=[target_group, category, weight, height, age],
        outputs=[output_size, output_conf]
    )

demo.launch()