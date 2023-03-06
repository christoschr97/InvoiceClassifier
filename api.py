from flask import Flask, jsonify, request
from transformers import pipeline

app = Flask(__name__)

classifier = pipeline(model="facebook/bart-large-mnli")

# Define the list of categories
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


@app.route('/', methods=['POST'])
def classify_invoice():
    if request.method == 'POST':
        # Get the invoice description from the request body
        data = request.get_json()
        description = data.get('description', None)
        if not description:
            return jsonify({'error': 'Invoice description not found in request.'}), 400

        # Perform zero-shot classification on the user-provided invoice description
        results = classifier(description, candidate_labels=categories)
        
        # Return the top predicted category and score
        return jsonify({
            'top_predicted_category': results['labels'][0],
            'score': results['scores'][0],
            'full_results': results
        }), 200


if __name__ == '__main__':
    app.run(debug=True)
