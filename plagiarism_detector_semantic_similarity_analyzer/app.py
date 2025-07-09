import streamlit as st
import numpy as np
import pandas as pd

from engine.embeddings import load_models
from engine.similarity import calculate_similarity_matrix
from engine.detection import detect_clones
from utils.visualization import create_heatmap

# Configure page
st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 Plagiarism Detector - Semantic Similarity Analyzer")
    st.markdown("---")
    
    # Load models
    models = load_models()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        selected_model_name = st.selectbox(
            "Choose Embedding Model:",
            options=list(models.keys()),
            help="Different models have different speed/accuracy tradeoffs"
        )
        
        threshold = st.slider(
            "Clone Detection Threshold (%)",
            min_value=50,
            max_value=95,
            value=80,
            help="Similarity percentage above which texts are considered clones"
        )
        
        num_texts = st.number_input(
            "Number of Texts to Compare",
            min_value=2,
            max_value=10,
            value=3
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Texts")
        texts = []
        
        for i in range(num_texts):
            text = st.text_area(
                f"Text {i+1}:",
                height=100,
                key=f"text_{i}",
                placeholder=f"Enter text {i+1} here..."
            )
            texts.append(text)
        
        analyze_button = st.button("🔍 Analyze Similarity", type="primary")
    
    with col2:
        st.header("📊 Results")
        
        if analyze_button:
            non_empty_texts = [text for text in texts if text.strip()]
            
            if len(non_empty_texts) < 2:
                st.error("Please enter at least 2 texts to compare!")
                return
            
            st.info(f"Using model: **{selected_model_name}**")
            
            with st.spinner("Calculating similarities..."):
                selected_model = models[selected_model_name]
                similarity_matrix = calculate_similarity_matrix(non_empty_texts, selected_model)
            
            st.subheader("Similarity Matrix")
            text_labels = [f"Text {i+1}" for i in range(len(non_empty_texts))]
            
            df = pd.DataFrame(
                similarity_matrix,
                index=text_labels,
                columns=text_labels
            )
            
            st.dataframe(df.style.format("{:.2f}%").background_gradient(cmap='RdYlBu_r'))
            
            clones = detect_clones(similarity_matrix, threshold)
            
            st.subheader("🚨 Clone Detection Results")
            if clones:
                st.error(f"Found {len(clones)} potential plagiarism cases:")
                
                for clone in clones:
                    st.write(f"**Text {clone['Text A']} ↔ Text {clone['Text B']}**: {clone['Similarity']:.2f}% similarity")
                    
                    with st.expander(f"View Text {clone['Text A']} vs Text {clone['Text B']}"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**Text {clone['Text A']}:**")
                            st.write(non_empty_texts[clone['Text A']-1])
                        with col_b:
                            st.write(f"**Text {clone['Text B']}:**")
                            st.write(non_empty_texts[clone['Text B']-1])
            else:
                st.success("No potential plagiarism detected!")
            # __________________________Visualization_________________
            # st.markdown("---")
            # st.header("📈 Visualization")
            # fig = create_heatmap(similarity_matrix, text_labels)
            # st.plotly_chart(fig, use_container_width=True)
            # ________________________________________________________
            
            st.markdown("---")
            st.header("⚖️ Model Comparison")
            
            if st.button("Compare All Models"):
                comparison_results = []
                
                for model_name, model in models.items():
                    sim_matrix = calculate_similarity_matrix(non_empty_texts, model)
                    detected_clones = detect_clones(sim_matrix, threshold)
                    
                    mask = np.ones_like(sim_matrix, dtype=bool)
                    np.fill_diagonal(mask, False)
                    avg_similarity = sim_matrix[mask].mean()
                    
                    comparison_results.append({
                        'Model': model_name,
                        'Average Similarity': f"{avg_similarity:.2f}%",
                        'Clones Detected': len(detected_clones),
                        'Max Similarity': f"{sim_matrix[mask].max():.2f}%"
                    })
                
                comparison_df = pd.DataFrame(comparison_results)
                st.dataframe(comparison_df, use_container_width=True)
    
    with st.expander("📚 How It Works"):
        st.markdown("""
        ### How Semantic Similarity Detection Works

        1. **Text Preprocessing**: Text is cleaned and normalized  
        2. **Embedding Generation**: Each text is converted to a high-dimensional vector representation  
        3. **Similarity Calculation**: Cosine similarity is computed between all text pairs  
        4. **Clone Detection**: Pairs exceeding the threshold are flagged as potential plagiarism  

        ### Embedding Models Explained

        - **MiniLM (Fast)**: Lightweight model, good for quick analysis  
        - **MPNet (Balanced)**: Better accuracy with reasonable speed  

        ### Why Embeddings Work for Plagiarism Detection

        - Embeddings capture semantic meaning, not just word matching  
        - Can detect paraphrasing and synonym usage  
        - Robust to minor text modifications  
        - Language-agnostic approach  
        """)

if __name__ == "__main__":
    main()
