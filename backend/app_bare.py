from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print(">>> STARTING BARE FLASK SERVER ON 5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
