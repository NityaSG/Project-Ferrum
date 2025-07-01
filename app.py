import streamlit as st
import io
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List
import json
import os
from dotenv import load_dotenv
load_dotenv()
# Configure Streamlit page for mobile optimization
st.set_page_config(
    page_title="🍎 Food Calorie Scanner",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile-friendly design
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.stCamera > div {
    display: flex;
    justify-content: center;
}
.food-item {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid #ff6b6b;
}
.nutrition-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.nutrition-item {
    text-align: center;
    background: white;
    padding: 0.5rem;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.total-calories {
    background: linear-gradient(90deg, #ff6b6b, #ffa500);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Pydantic models for structured output
class FoodItem(BaseModel):
    name: str
    portion_grams: float
    protein_grams: float
    calories: float
    carbs_grams: float

class NutritionAnalysis(BaseModel):
    food_items: List[FoodItem]
    total_calories: float
    confidence_level: str

# Initialize Gemini client
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

def analyze_food_image(image_bytes: bytes) -> NutritionAnalysis:
    """Analyze food image using Gemini and return structured nutrition data"""
    try:
        client = get_gemini_client()
        
        prompt = """
        Analyze this food image and provide detailed nutritional information. 
        Identify each food item visible in the image and estimate:
        1. The name of each food item
        2. The portion size in grams
        3. Protein content in grams
        4. Calories
        5. Carbohydrates in grams
        
        Be as accurate as possible with portion estimation based on visual cues like plate size, 
        utensils, or common serving sizes. Provide a confidence level (high/medium/low) for your analysis.
        
        If multiple food items are present, list each separately.
        Calculate the total calories for the entire meal.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                prompt
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": NutritionAnalysis,
            }
        )
        
        return response.parsed
    
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

def display_nutrition_results(analysis: NutritionAnalysis):
    """Display nutrition analysis results in a mobile-friendly format"""
    
    # Total calories display
    st.markdown(f"""
    <div class="total-calories">
        🔥 Total Calories: {analysis.total_calories:.0f} kcal
        <br><small>Confidence: {analysis.confidence_level}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Individual food items
    st.subheader("📋 Food Items Detected")
    
    for i, item in enumerate(analysis.food_items, 1):
        st.markdown(f"""
        <div class="food-item">
            <h4>🍽️ {item.name}</h4>
            <p><strong>Portion:</strong> {item.portion_grams:.0f}g</p>
            <div class="nutrition-grid">
                <div class="nutrition-item">
                    <div style="font-size: 1.2rem; color: #ff6b6b;">🔥</div>
                    <div style="font-weight: bold;">{item.calories:.0f}</div>
                    <div style="font-size: 0.8rem;">Calories</div>
                </div>
                <div class="nutrition-item">
                    <div style="font-size: 1.2rem; color: #4ecdc4;">💪</div>
                    <div style="font-weight: bold;">{item.protein_grams:.1f}g</div>
                    <div style="font-size: 0.8rem;">Protein</div>
                </div>
                <div class="nutrition-item">
                    <div style="font-size: 1.2rem; color: #45b7d1;">🌾</div>
                    <div style="font-weight: bold;">{item.carbs_grams:.1f}g</div>
                    <div style="font-size: 0.8rem;">Carbs</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main app
def main():
    st.title("🍎 Food Calorie Scanner")
    st.markdown("📱 *Take a photo of your food to get instant nutrition analysis*")
    
    # API Key input (for development)
    # if "GEMINI_API_KEY" not in st.secrets:
    #     api_key = st.text_input(
    #         "🔑 Enter your Gemini API Key:",
    #         type="password",
    #         help="Get your API key from Google AI Studio"
    #     )
    #     if api_key:
    #         st.session_state["gemini_api_key"] = api_key
    
    # Camera input
    st.subheader("📸 Capture Your Food")
    
    # Instructions
    with st.expander("📋 How to use", expanded=False):
        st.markdown("""
        1. **Position your food** - Make sure all food items are clearly visible
        2. **Good lighting** - Ensure adequate lighting for better analysis
        3. **Include reference** - Objects like utensils help with portion estimation
        4. **Take photo** - Click the camera button below
        5. **Get results** - AI will analyze and provide nutrition information
        """)
    
    # Camera widget
    camera_image = st.camera_input("📷 Take a picture of your food")
    
    # Alternative: File upload for testing
    st.markdown("*Or upload an image:*")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    # Process image
    image_source = camera_image or uploaded_file
    
    if image_source is not None:
        # Display the image
        st.subheader("🖼️ Your Food Photo")
        image = Image.open(image_source)
        st.image(image, caption="Food to analyze", use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Nutrition", type="primary", use_container_width=True):
            
            # Check for API key
            # if "GEMINI_API_KEY" not in st.secrets and "gemini_api_key" not in st.session_state:
            #     st.error("⚠️ Please enter your Gemini API key to proceed")
            #     return
            
            with st.spinner("🤖 AI is analyzing your food..."):
                # Convert image to bytes
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='JPEG')
                img_bytes = img_bytes.getvalue()
                
                # Analyze with Gemini
                analysis = analyze_food_image(img_bytes)
                
                if analysis:
                    st.success("✅ Analysis complete!")
                    display_nutrition_results(analysis)
                    
                    # Save to session state for later reference
                    st.session_state["last_analysis"] = analysis
                else:
                    st.error("❌ Failed to analyze the image. Please try again.")
    
    # Show previous analysis if available
    if "last_analysis" in st.session_state and image_source is None:
        st.subheader("📊 Last Analysis")
        display_nutrition_results(st.session_state["last_analysis"])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <small>
        🤖 Powered by Google Gemini AI<br>
        📱 Optimized for mobile use<br>
        ⚠️ Results are estimates - consult professionals for precise dietary needs
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()