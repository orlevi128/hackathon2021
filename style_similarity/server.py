import os
from flask import Flask, request, redirect, flash, render_template
from werkzeug.utils import secure_filename
from config import UPLOAD_DIR
import recommender

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.secret_key = "best key of secrecy ever"


@app.route('/', methods=['GET', 'POST'])
def serve():
    if request.method == 'POST':
        number_of_recommendations = 20 if 'rec_num' not in request.form else int(request.form['rec_num'])
        print(request.files)
        if 'img' not in request.files:
            flash('No file part')
            return redirect(request.url)
        img = request.files['img']
        if img.filename == '':
            flash('No selected file')
            return redirect(request.url)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img.filename))
        img.save(img_path)
        recommender.match_recommendations(img_path, number_of_recommendations)
        return str({'response': 'OK'}).replace('\'', '"')
    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print('Starting server')
    app.debug = True
    app.run(host='0.0.0.0', port=port)
