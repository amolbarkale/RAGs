import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🔍",
    layout="wide"
)

def load_models():
    """Load different embedding models"""
    try:
        models = {
            'MiniLM (Fast)': SentenceTransformer('all-MiniLM-L6-v2'),
            'MPNet (Balanced)': SentenceTransformer('all-mpnet-base-v2'),
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {'MiniLM (Fast)': SentenceTransformer('all-MiniLM-L6-v2')}

def preprocess_text(text):
    """Simple text preprocessing"""
    return text.strip().lower()

def calculate_similarity_matrix(texts, model):
    """Calculate pairwise similarity matrix"""
    processed_texts = [preprocess_text(text) for text in texts]
    embeddings = model.encode(processed_texts)
    similarity_matrix = cosine_similarity(embeddings)
    similarity_percentage = (similarity_matrix * 100).round(2)
    return similarity_percentage

def detect_clones(similarity_matrix, threshold=80):
    """Detect potential clones based on similarity threshold"""
    clones = []
    n = len(similarity_matrix)
    
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i][j] >= threshold:
                clones.append({
                    'Text A': i+1,
                    'Text B': j+1,
                    'Similarity': similarity_matrix[i][j]
                })
    
    return clones

def create_heatmap(similarity_matrix, text_labels):
    """Create interactive heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=text_labels,
        y=text_labels,
        colorscale='RdYlBu_r',
        text=similarity_matrix,
        texttemplate='%{text}%',
        textfont={"size": 12},
        colorbar=dict(title="Similarity %")
    ))
    
    fig.update_layout(
        title="Similarity Matrix Heatmap",
        xaxis_title="Texts",
        yaxis_title="Texts",
        width=600,
        height=500
    )
    
    return fig

def main():
    st.title("🔍 Plagiarism Detector - Semantic Similarity Analyzer")
    st.markdown("---")
    
    # Load models
    models = load_models()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        selected_model_name = st.selectbox(
            "Choose Embedding Model:",
            options=list(models.keys()),
            help="Different models have different speed/accuracy tradeoffs"
        )
        
        # Threshold setting
        threshold = st.slider(
            "Clone Detection Threshold (%)",
            min_value=50,
            max_value=95,
            value=80,
            help="Similarity percentage above which texts are considered clones"
        )
        
        # Number of texts
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
        
        # Dynamic text input boxes
        for i in range(num_texts):
            text = st.text_area(
                f"Text {i+1}:",
                height=100,
                key=f"text_{i}",
                placeholder=f"Enter text {i+1} here..."
            )
            texts.append(text)
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Similarity", type="primary")
    
    with col2:
        st.header("📊 Results")
        
        if analyze_button:
            # Validate inputs
            non_empty_texts = [text for text in texts if text.strip()]
            
            if len(non_empty_texts) < 2:
                st.error("Please enter at least 2 texts to compare!")
                return
            
            # Show selected model info
            st.info(f"Using model: **{selected_model_name}**")
            
            # Calculate similarity
            with st.spinner("Calculating similarities..."):
                selected_model = models[selected_model_name]
                similarity_matrix = calculate_similarity_matrix(non_empty_texts, selected_model)
            
            # Display similarity matrix
            st.subheader("Similarity Matrix")
            
            # Create labels
            text_labels = [f"Text {i+1}" for i in range(len(non_empty_texts))]
            
            # Create DataFrame for display
            df = pd.DataFrame(
                similarity_matrix,
                index=text_labels,
                columns=text_labels
            )
            
            # Display table
            st.dataframe(df.style.format("{:.2f}%").background_gradient(cmap='RdYlBu_r'))
            
            # Clone detection
            clones = detect_clones(similarity_matrix, threshold)
            
            st.subheader("🚨 Clone Detection Results")
            if clones:
                st.error(f"Found {len(clones)} potential plagiarism cases:")
                
                for clone in clones:
                    st.write(f"**Text {clone['Text A']} ↔ Text {clone['Text B']}**: {clone['Similarity']:.2f}% similarity")
                    
                    # Show the actual texts
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
            
            # Visualization
            st.markdown("---")
            st.header("📈 Visualization")
            
            # Create heatmap
            fig = create_heatmap(similarity_matrix, text_labels)
            st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison
            st.markdown("---")
            st.header("⚖️ Model Comparison")
            
            if st.button("Compare All Models"):
                comparison_results = []
                
                for model_name, model in models.items():
                    sim_matrix = calculate_similarity_matrix(non_empty_texts, model)
                    detected_clones = detect_clones(sim_matrix, threshold)
                    
                    # Calculate average similarity (excluding diagonal)
                    mask = np.ones_like(sim_matrix, dtype=bool)
                    np.fill_diagonal(mask, False)
                    avg_similarity = sim_matrix[mask].mean()
                    
                    comparison_results.append({
                        'Model': model_name,
                        'Average Similarity': f"{avg_similarity:.2f}%",
                        'Clones Detected': len(detected_clones),
                        'Max Similarity': f"{sim_matrix[mask].max():.2f}%"
                    })
                
                # Display comparison table
                comparison_df = pd.DataFrame(comparison_results)
                st.dataframe(comparison_df, use_container_width=True)
    
    # Documentation section
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