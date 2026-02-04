from flask import Flask, render_template, request
import os
# Importă funcția de clasificare din scriptul tău
# Presupunem că ai o funcție numită 'predict' în clasifica.py
from clasifica import predict 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Nu ai selectat nicio imagine"
        
        file = request.files['file']
        if file.filename == '':
            return "Nume fisier invalid"

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Aici apelezi funcția ta de clasificare
            rezultat = predict(filepath) 
            
            return f"Rezultatul clasificării: {rezultat}"

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)