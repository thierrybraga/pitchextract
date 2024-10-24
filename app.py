from flask import Flask, request, redirect, url_for, render_template, flash
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from pydub import AudioSegment
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CONVERTED_FOLDER'] = 'converted'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac'}
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # 128 MB
app.config['SECRET_KEY'] = 'supersecretkey'  # Necessário para flash messages

# Criação dos diretórios necessários, se não existirem
for folder in [app.config['UPLOAD_FOLDER'], app.config['CONVERTED_FOLDER'], 'static/spectrograms']:
    if not os.path.exists(folder):
        os.makedirs(folder)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Frequency Spectrogram')
    plt.tight_layout()

    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close()
    image_stream.seek(0)

    return image_stream


def compute_mfcc(file_path, n_mfcc=15, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    y = librosa.effects.preemphasis(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                 hop_length=512, n_fft=2048,
                                 fmin=0, fmax=sr / 2)
    mfccs -= np.mean(mfccs, axis=1, keepdims=True)
    mfccs /= np.std(mfccs, axis=1, keepdims=True)

    return mfccs


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Nenhum arquivo foi selecionado.', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                # Convert to FLAC
                flac_filename = f"{os.path.splitext(filename)[0]}.flac"
                converted_path = os.path.join(app.config['CONVERTED_FOLDER'], flac_filename)
                audio = AudioSegment.from_file(file_path)
                audio.export(converted_path, format='flac')

                # Remove original file
                os.remove(file_path)

                # Compute MFCC
                mfccs = compute_mfcc(converted_path)

                # Plot Spectrogram
                spectrogram_image = plot_spectrogram(converted_path)
                spectrogram_path = os.path.join('static', 'spectrograms', f'{os.path.splitext(flac_filename)[0]}.png')
                with open(spectrogram_path, 'wb') as f:
                    f.write(spectrogram_image.read())

                return render_template('index.html',
                                       spectrogram_url=url_for('static',
                                                               filename=f'spectrograms/{os.path.splitext(flac_filename)[0]}.png'),
                                       mfccs=mfccs)
            except Exception as e:
                flash(f'Ocorreu um erro ao processar o arquivo: {str(e)}', 'error')
                return redirect(request.url)

        else:
            flash('Tipo de arquivo não permitido.', 'error')
            return redirect(request.url)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
