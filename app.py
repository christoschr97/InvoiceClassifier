import torch
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


def load_models():
    classifier = pipeline(model="facebook/bart-large-mnli")
    categories = [
      'Products and Services',
      'Subscription Fees',
      'Rent and Leases',
      'Utilities and Expenses',
      'Membership and Dues',
      'Insurance',
      'Taxes',
      'Donations and Contributions',
      'Interest and Late Fees',
      'Miscellaneous'
    ]
    return classifier, categories

@st.cache(allow_output_mutation=True)
def get_models():
    return load_models()


def app():
    # Set the app title
    st.title('Invoice Category Classifier')
    
    # Ask the user for the invoice description
    description = st.text_input('Invoice Description')
    
    # Get the classifier and categories from the cached models
    classifier, categories = get_models()
    
    # Perform zero-shot classification on the user-provided invoice description
    if description:
        results = classifier(description, candidate_labels=categories)
        
        # Display the top predicted category and score
        st.write(f'Top predicted category: {results["labels"][0]}')
        st.write(f'With score: {results["scores"][0]}')
        st.write(f'Full results: {results}')


if __name__ == '__main__':
    app()
