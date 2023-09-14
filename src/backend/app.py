from flask import Flask, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/message', methods=['GET'])
@cross_origin()
def get_message():
    return jsonify({'message': 'Hello from the backend!'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)