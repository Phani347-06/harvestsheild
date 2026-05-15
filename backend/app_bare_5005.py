from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print(">>> STARTING BARE FLASK SERVER ON 5005")
    app.run(host='127.0.0.1', port=5005, debug=False)
