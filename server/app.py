from flask import Flask, request, jsonify
import os
from store_csv import store_csv_in_chroma

app = Flask(__name__)

@app.route('/json', methods=['POST'])
def receive_json():
    data = request.get_json()
    print('JSON recibido:', data)
    return jsonify({'mensaje': 'JSON recibido correctamente', 'recibido': data})

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    save_path = os.path.join(os.getcwd(), file.filename)
    file.save(save_path)
    # Procesar y almacenar en ChromaDB
    try:
        store_csv_in_chroma(save_path)
        return jsonify({'mensaje': 'CSV recibido y almacenado en ChromaDB', 'archivo': file.filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
