from flask import Flask, jsonify, request  # Add 'request' here
from flask_cors import CORS, cross_origin
import pandas as pd    
import requests

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load your DataFrame here
df = pd.read_csv("papers.csv")

@app.route('/api/message', methods=['GET'])
@cross_origin()
def get_papers():
    title_filter = request.args.get('title')
    
    if title_filter:
        filtered_df = df[df['Title'].str.contains(title_filter, case=False)]
    else:
        filtered_df = df
    
    return jsonify(filtered_df.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)