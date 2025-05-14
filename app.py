import pickle
import os
import spacy
import streamlit as st
from strip_polite import remove_polite_features

def lemmatize_text(text, nlp):
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc])

def load_resources():

    if not os.path.exists('model/polite_features.pkl'):
        st.error("Error: Model files not found. Please run main.py first to train the model.")
        st.stop()
        
    with open('model/polite_features.pkl', 'rb') as f:
        polite_features_dict = pickle.load(f)
    
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        st.error("Error: spaCy model 'en_core_web_sm' not found. Please install it with:")
        st.code("python -m spacy download en_core_web_sm")
        st.stop()
    
    return polite_features_dict, nlp

def main():

    st.title("Polite Language Remover")
    st.write("This app removes polite words and phrases from your text using machine learning with word n-grams and lemmatization.")
    
    with st.spinner("Loading resources..."):
        polite_features_dict, nlp = load_resources()
    
    st.success(f"Loaded {len(polite_features_dict)} polite features.")
    
    threshold = st.sidebar.slider(
        "Politeness Threshold", 
        min_value=0.0, 
        max_value=2.0, 
        value=0.05,
        step=0.01,
        help="Higher values = fewer features removed. Lower values = more features removed."
    )
    
    feature_type = st.sidebar.radio(
        "Feature types to remove:",
        ["All features", "Words only", "N-grams only"],
        help="Choose which types of polite features to remove."
    )
    
    if st.sidebar.checkbox("Show top polite features"):
        sorted_features = sorted(polite_features_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Group features by type
        words = [(f, s) for f, s in sorted_features if ' ' not in f]
        bigrams = [(f, s) for f, s in sorted_features if ' ' in f and f.count(' ') == 1]
        trigrams = [(f, s) for f, s in sorted_features if f.count(' ') >= 2]
        
        # Show top features by type
        if st.sidebar.checkbox("Single words"):
            st.sidebar.write("Top 15 polite words:")
            for feature, score in words[:15]:
                st.sidebar.write(f"{feature}: {score:.4f}")
                
        if st.sidebar.checkbox("Bigrams (2-word phrases)"):
            st.sidebar.write("Top 15 polite bigrams:")
            for feature, score in bigrams[:15]:
                st.sidebar.write(f"{feature}: {score:.4f}")
                
        if st.sidebar.checkbox("Trigrams (3-word phrases)"):
            st.sidebar.write("Top 15 polite trigrams:")
            for feature, score in trigrams[:15]:
                st.sidebar.write(f"{feature}: {score:.4f}")
    
    # Text input
    user_input = st.text_area("Enter your text:", height=150)
    
    # Process when user enters text
    if user_input:
        # Filter based on user selection
        filtered_features = {}
        
        if feature_type == "All features":
            filtered_features = polite_features_dict
        elif feature_type == "Words only":
            filtered_features = {f: s for f, s in polite_features_dict.items() if ' ' not in f}
        elif feature_type == "N-grams only":
            filtered_features = {f: s for f, s in polite_features_dict.items() if ' ' in f}
        
        filtered_text, removed_features = remove_polite_features(user_input, filtered_features, nlp, threshold)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Text")
            st.write(user_input)
            
        with col2:
            st.subheader("Filtered Text")
            st.write(filtered_text)
        
        # Show removed features
        if removed_features:
            st.subheader("Removed Features")
            st.write(", ".join(removed_features))
            
            # Show count of features removed
            st.info(f"Removed {len(removed_features)} polite feature(s)")
            
            # Highlight the differences
            st.subheader("Text Comparison")
            highlighted_text = user_input
            for feature in sorted(removed_features, key=len, reverse=True):
                if feature in highlighted_text:
                    highlighted_text = highlighted_text.replace(feature, f"**{feature}**")
            st.markdown(f"Original text with removed features highlighted: \n\n{highlighted_text}")
        else:
            st.info("No polite features were removed.")
            
        # Add a comparison of politeness score
        if st.checkbox("Show politeness analysis"):
            st.subheader("Politeness Analysis")
            
            # Load the model to get politeness predictions
            try:
                with open('model/polite_classifier.pkl', 'rb') as f:
                    model = pickle.load(f)
                    
                # Get prediction probabilities for original and filtered text
                original_proba = model.predict_proba([lemmatize_text(user_input, nlp)])[0][1]  # Probability of being polite
                filtered_proba = model.predict_proba([lemmatize_text(filtered_text, nlp)])[0][1]
                
                # Display probabilities
                st.write(f"Original text politeness score: {original_proba:.2f}")
                st.write(f"Filtered text politeness score: {filtered_proba:.2f}")
                
                # Calculate reduction percentage
                if original_proba > 0:
                    reduction = ((original_proba - filtered_proba) / original_proba) * 100
                    st.write(f"Politeness reduced by: {reduction:.1f}%")
            except Exception as e:
                st.error(f"Could not load model for politeness analysis: {str(e)}")

if __name__ == "__main__":
    main() 
