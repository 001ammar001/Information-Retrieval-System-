from flask import Flask, request, jsonify
from flask_cors import CORS
from app.services.search_service import search_service
from app.online.query_suggestion_service import querySuggestion
from app.services.text_processing_services import text_processing_service

app = Flask(__name__)
CORS(app)

def validate_query_parameters(data_set_name, query):
    if not data_set_name:
        return {'error': 'data_set_name parameter is missing'}, 400
    if not query:
        return {'error': 'query parameter is missing'}, 400

    if data_set_name not in ['antique', 'quora']:
        return {'error': 'Invalid data_set_name. Must be "antique" or "quora"'}, 400
    
    return None, None

@app.route('/get_user_query/main_feature', methods=['GET'])
def get_main_feature_query():
    data_set_name = request.args.get('data_set_name')
    query = request.args.get('query')
    top_k = int(request.args.get('top_k', 5))  # Default to 5 results
    method = request.args.get('method', 'tfidf')  # Default to tfidf

    error_response, status_code = validate_query_parameters(data_set_name, query)
    if error_response:
        return jsonify(error_response), status_code

    try:
        results = search_service.search(query, data_set_name, top_k, method)
        return jsonify(results)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    
@app.route('/get_search_suggestion',methods=['GET'])
def get_search_suggestion():
    data_set_name = request.args.get('data_set_name')
    query = request.args.get('query')   
    error_response, status_code = validate_query_parameters(data_set_name, query)
    if error_response:
        return jsonify(error_response), status_code
    
    try:
        results = querySuggestion.suggest(query)
        return jsonify(results)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    

@app.route('/get_user_query/secondary_features', methods=['GET'])
def get_secondary_features_query():
    data_set_name = request.args.get('data_set_name')
    query = request.args.get('query')
    top_k = int(request.args.get('top_k', 5))  # Default to 5 results
    method = request.args.get('method', 'tfidf')  # Default to tfidf

    error_response, status_code = validate_query_parameters(data_set_name, query)
    if error_response:
        return jsonify(error_response), status_code

    try:
        # For secondary features, we could implement different search logic
        # For now, using the same cosine similarity search
        results = search_service.search(query, data_set_name, top_k, method)
        return jsonify(results)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/process_text', methods=['GET'])
def process_text_endpoint():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'query parameter is missing'}), 400
    processed = text_processing_service.process_text(query)
    return jsonify({'processed_query': processed})

if __name__ == '__main__':
    app.run()