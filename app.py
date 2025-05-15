import pickle
import streamlit as st
from strip_polite import remove_polite_features
from spell_correct import correct_spell
from utils import load_resources, lemmatize_text

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
        
        words = [(f, s) for f, s in sorted_features if ' ' not in f]
        bigrams = [(f, s) for f, s in sorted_features if ' ' in f and f.count(' ') == 1]
        trigrams = [(f, s) for f, s in sorted_features if f.count(' ') >= 2]
        
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
    
    user_input = st.text_area("Enter your text:", height=150)
    
    if user_input:
        corrected_text = correct_spell(user_input)
        
        if corrected_text != user_input:
            st.info("Text has been spell-corrected.")
            
        filtered_features = {}
        
        if feature_type == "All features":
            filtered_features = polite_features_dict
        elif feature_type == "Words only":
            filtered_features = {f: s for f, s in polite_features_dict.items() if ' ' not in f}
        elif feature_type == "N-grams only":
            filtered_features = {f: s for f, s in polite_features_dict.items() if ' ' in f}
        
        filtered_text, removed_features = remove_polite_features(corrected_text, filtered_features, nlp, threshold)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Text")
            st.write(user_input)
            
            if corrected_text != user_input:
                st.subheader("Spell-corrected Text")
                st.write(corrected_text)
            
        with col2:
            st.subheader("Filtered Text")
            st.write(filtered_text)
        
        if removed_features:
            st.subheader("Removed Features")
            st.write(", ".join(removed_features))
            
            
            st.info(f"Removed {len(removed_features)} polite feature(s)")
            
            st.subheader("Text Comparison")
            highlighted_text = corrected_text
            for feature in sorted(removed_features, key=len, reverse=True):
                if feature in highlighted_text:
                    highlighted_text = highlighted_text.replace(feature, f"**{feature}**")
            st.markdown(f"Text with removed features highlighted: \n\n{highlighted_text}")
        else:
            st.info("No polite features were removed.")
            
        if st.checkbox("Show politeness analysis"):
            st.subheader("Politeness Analysis")
            
            try:
                with open('model/polite_classifier.pkl', 'rb') as f:
                    model = pickle.load(f)
                    
                original_proba = model.predict_proba([lemmatize_text(corrected_text, nlp)])[0][1]  # Probability of being polite
                filtered_proba = model.predict_proba([lemmatize_text(filtered_text, nlp)])[0][1]
                
                st.write(f"Original text politeness score: {original_proba:.2f}")
                st.write(f"Filtered text politeness score: {filtered_proba:.2f}")
                
                if original_proba > 0:
                    reduction = ((original_proba - filtered_proba) / original_proba) * 100
                    st.write(f"Politeness reduced by: {reduction:.1f}%")
            except Exception as e:
                st.error(f"Could not load model for politeness analysis: {str(e)}")

if __name__ == "__main__":
    main() 
