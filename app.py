"""
Streamlit Web Interface for Advanced POS Tagger
===============================================

A modern, interactive web interface for the Part-of-Speech tagger.
Features real-time analysis, model comparison, and data visualization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from typing import Dict, List

# Import our POS tagger
from pos_tagger import AdvancedPOSTagger, SentenceAnalysis

# Page configuration
st.set_page_config(
    page_title="Advanced POS Tagger",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_tagger():
    """Load the POS tagger with caching"""
    return AdvancedPOSTagger()

def create_pos_distribution_chart(analysis: SentenceAnalysis):
    """Create POS tag distribution chart"""
    pos_counts = pd.Series([token.pos_tag for token in analysis.tokens]).value_counts()
    
    fig = px.pie(
        values=pos_counts.values,
        names=pos_counts.index,
        title="POS Tag Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_confidence_chart(analysis: SentenceAnalysis):
    """Create confidence scores chart"""
    tokens = [token.token for token in analysis.tokens]
    confidences = [token.confidence for token in analysis.tokens]
    
    fig = go.Figure(data=[
        go.Bar(
            x=tokens,
            y=confidences,
            marker_color='lightblue',
            text=[f"{c:.3f}" for c in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores by Token",
        xaxis_title="Tokens",
        yaxis_title="Confidence Score",
        xaxis_tickangle=-45
    )
    
    return fig

def create_model_comparison_chart(comparisons: Dict[str, SentenceAnalysis]):
    """Create model comparison chart"""
    models = list(comparisons.keys())
    processing_times = [analysis.processing_time for analysis in comparisons.values()]
    confidence_scores = [analysis.confidence_score for analysis in comparisons.values()]
    token_counts = [len(analysis.tokens) for analysis in comparisons.values()]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Processing Time", "Confidence Score", "Token Count"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Processing time
    fig.add_trace(
        go.Bar(x=models, y=processing_times, name="Processing Time", marker_color='lightcoral'),
        row=1, col=1
    )
    
    # Confidence score
    fig.add_trace(
        go.Bar(x=models, y=confidence_scores, name="Confidence Score", marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Token count
    fig.add_trace(
        go.Bar(x=models, y=token_counts, name="Token Count", marker_color='lightblue'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="Model Comparison")
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Advanced Part-of-Speech Tagger</h1>', unsafe_allow_html=True)
    
    # Load tagger
    with st.spinner("Loading POS tagger models..."):
        tagger = load_tagger()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    available_models = list(tagger.models.keys())
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        index=0
    )
    
    # Sample text selection
    st.sidebar.header("Sample Texts")
    sample_data = tagger.db.get_sample_texts(limit=20)
    sample_options = {f"{category} (Level {difficulty})": text for _, category, text, difficulty in sample_data}
    
    selected_sample = st.sidebar.selectbox(
        "Choose Sample Text",
        ["Custom Input"] + list(sample_options.keys())
    )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Analysis", "📊 Model Comparison", "📈 Batch Processing", "🗄️ Database Explorer"])
    
    with tab1:
        st.header("Single Text Analysis")
        
        # Text input
        if selected_sample == "Custom Input":
            text_input = st.text_area(
                "Enter text to analyze:",
                value="The quick brown fox jumps over the lazy dog.",
                height=100
            )
        else:
            text_input = sample_options[selected_sample]
            st.text_area("Selected text:", value=text_input, height=100, disabled=True)
        
        # Analysis button
        if st.button("Analyze Text", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    start_time = time.time()
                    analysis = tagger.tag_text(text_input, selected_model)
                    processing_time = time.time() - start_time
                
                # Display results
                st.success(f"Analysis completed in {processing_time:.4f} seconds!")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Processing Time", f"{analysis.processing_time:.4f}s")
                with col2:
                    st.metric("Average Confidence", f"{analysis.confidence_score:.3f}")
                with col3:
                    st.metric("Token Count", len(analysis.tokens))
                with col4:
                    st.metric("Model", analysis.model_name)
                
                # Results table
                st.subheader("POS Tagging Results")
                results_data = []
                for token in analysis.tokens:
                    results_data.append({
                        "Token": token.token,
                        "POS Tag": token.pos_tag,
                        "Tag": token.tag,
                        "Explanation": token.explanation,
                        "Confidence": f"{token.confidence:.3f}"
                    })
                
                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_pos_distribution_chart(analysis), use_container_width=True)
                
                with col2:
                    st.plotly_chart(create_confidence_chart(analysis), use_container_width=True)
                
                # Save to database
                sample_texts = tagger.db.get_sample_texts(limit=1)
                if sample_texts:
                    text_id, _, _, _ = sample_texts[0]
                    tagger.db.save_results(text_id, analysis.model_name, analysis.tokens, 
                                         analysis.processing_time, analysis.confidence_score)
                    st.info("Results saved to database!")
            else:
                st.error("Please enter some text to analyze.")
    
    with tab2:
        st.header("Model Comparison")
        
        comparison_text = st.text_area(
            "Text for model comparison:",
            value="The sophisticated algorithm efficiently processes natural language data.",
            height=100
        )
        
        if st.button("Compare Models", type="primary"):
            if comparison_text.strip():
                with st.spinner("Comparing models..."):
                    comparisons = tagger.compare_models(comparison_text)
                
                if comparisons:
                    st.success(f"Compared {len(comparisons)} models successfully!")
                    
                    # Comparison chart
                    st.plotly_chart(create_model_comparison_chart(comparisons), use_container_width=True)
                    
                    # Detailed comparison table
                    st.subheader("Detailed Comparison")
                    comparison_data = []
                    for model_name, analysis in comparisons.items():
                        comparison_data.append({
                            "Model": model_name,
                            "Processing Time (s)": f"{analysis.processing_time:.4f}",
                            "Confidence Score": f"{analysis.confidence_score:.3f}",
                            "Token Count": len(analysis.tokens),
                            "Text Length": len(analysis.text)
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Individual model results
                    st.subheader("Individual Model Results")
                    for model_name, analysis in comparisons.items():
                        with st.expander(f"Results from {model_name}"):
                            results_data = []
                            for token in analysis.tokens:
                                results_data.append({
                                    "Token": token.token,
                                    "POS Tag": token.pos_tag,
                                    "Tag": token.tag,
                                    "Confidence": f"{token.confidence:.3f}"
                                })
                            
                            df = pd.DataFrame(results_data)
                            st.dataframe(df, use_container_width=True)
                else:
                    st.error("No models available for comparison.")
            else:
                st.error("Please enter some text to compare.")
    
    with tab3:
        st.header("Batch Processing")
        
        # Batch text input
        batch_texts = st.text_area(
            "Enter multiple texts (one per line):",
            value="The quick brown fox jumps over the lazy dog.\nMachine learning is fascinating.\nNatural language processing helps computers understand text.",
            height=200
        )
        
        batch_model = st.selectbox("Model for batch processing:", available_models)
        
        if st.button("Process Batch", type="primary"):
            if batch_texts.strip():
                texts = [text.strip() for text in batch_texts.split('\n') if text.strip()]
                
                if texts:
                    with st.spinner(f"Processing {len(texts)} texts..."):
                        batch_results = tagger.batch_process(texts, batch_model)
                    
                    st.success(f"Processed {len(batch_results)} texts successfully!")
                    
                    # Batch results summary
                    summary_data = []
                    for i, result in enumerate(batch_results):
                        summary_data.append({
                            "Text #": i + 1,
                            "Text": result.text[:50] + "..." if len(result.text) > 50 else result.text,
                            "Tokens": len(result.tokens),
                            "Processing Time (s)": f"{result.processing_time:.4f}",
                            "Confidence": f"{result.confidence_score:.3f}"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Batch visualization
                    processing_times = [result.processing_time for result in batch_results]
                    confidence_scores = [result.confidence_score for result in batch_results]
                    token_counts = [len(result.tokens) for result in batch_results]
                    
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=("Processing Time", "Confidence Score", "Token Count"),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=list(range(1, len(processing_times) + 1)), y=processing_times, 
                                 mode='lines+markers', name="Processing Time"),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=list(range(1, len(confidence_scores) + 1)), y=confidence_scores, 
                                 mode='lines+markers', name="Confidence Score"),
                        row=1, col=2
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=list(range(1, len(token_counts) + 1)), y=token_counts, 
                                 mode='lines+markers', name="Token Count"),
                        row=1, col=3
                    )
                    
                    fig.update_layout(height=400, showlegend=False, title_text="Batch Processing Results")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No valid texts found.")
            else:
                st.error("Please enter some texts to process.")
    
    with tab4:
        st.header("Database Explorer")
        
        # Database statistics
        sample_data = tagger.db.get_sample_texts(limit=100)
        
        if sample_data:
            st.subheader("Sample Texts in Database")
            
            # Category filter
            categories = list(set([category for _, category, _, _ in sample_data]))
            selected_category = st.selectbox("Filter by category:", ["All"] + categories)
            
            # Filter data
            if selected_category == "All":
                filtered_data = sample_data
            else:
                filtered_data = [(id, cat, text, diff) for id, cat, text, diff in sample_data if cat == selected_category]
            
            # Display sample texts
            for text_id, category, text, difficulty in filtered_data[:10]:  # Show first 10
                with st.expander(f"{category.title()} (Level {difficulty}) - ID: {text_id}"):
                    st.write(text)
                    
                    # Show analysis button for each text
                    if st.button(f"Analyze with {selected_model}", key=f"analyze_{text_id}"):
                        with st.spinner("Analyzing..."):
                            analysis = tagger.tag_text(text, selected_model)
                        
                        st.write("**Results:**")
                        results_data = []
                        for token in analysis.tokens:
                            results_data.append({
                                "Token": token.token,
                                "POS Tag": token.pos_tag,
                                "Tag": token.tag,
                                "Confidence": f"{token.confidence:.3f}"
                            })
                        
                        df = pd.DataFrame(results_data)
                        st.dataframe(df, use_container_width=True)
            
            # Database statistics
            st.subheader("Database Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Texts", len(sample_data))
            
            with col2:
                category_counts = pd.Series([cat for _, cat, _, _ in sample_data]).value_counts()
                st.metric("Categories", len(category_counts))
            
            with col3:
                avg_difficulty = np.mean([diff for _, _, _, diff in sample_data])
                st.metric("Avg Difficulty", f"{avg_difficulty:.1f}")
            
            # Category distribution
            category_counts = pd.Series([cat for _, cat, _, _ in sample_data]).value_counts()
            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Text Distribution by Category",
                labels={"x": "Category", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No sample texts found in database.")

if __name__ == "__main__":
    main()
