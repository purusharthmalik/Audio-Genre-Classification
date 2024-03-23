#importing libraries
import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import librosa
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

st.set_page_config(layout="wide")

#markdown
st.markdown('<h1 style="color:black;">Audio Genre Classification</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color:black;">Our current model classifies a music file as belonging to one of the following genres:</h2>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
c1.markdown('<ul style="color:black;"> <li>Blues</li><li>Classical</li><li>Country</li><li>Disco</li><li>Hiphop</li></ul>', unsafe_allow_html=True)
c2.markdown('<ul style="color:black;"> <li>Jazz</li><li>Metal</li><li>Pop</li><li>Reggae</li><li>Rock</li></ul>', unsafe_allow_html=True)

# background image to streamlit

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('./canvas.png')

# Getting audio features
def audio_features(filename):
    df = pd.read_csv("GTZAN/features_30_sec.csv")
    data_max = np.array([7.49480784e-01, 1.20964289e-01, 4.42566752e-01, 3.26152220e-02,
        5.43253441e+03, 4.79411860e+06, 3.70814755e+03, 1.23514251e+06,
        9.48744648e+03, 1.29832038e+07, 3.47705078e-01, 6.51853790e-02,
        1.56880375e-02, 1.27051502e-01, 6.81878207e-03, 5.88790923e-02,
        2.87109375e+02, 1.07941315e+02, 4.50273750e+04, 2.51212494e+02,
        5.13198730e+03, 8.08459244e+01, 4.14778613e+03, 8.97173386e+01,
        2.30375195e+03, 4.68342056e+01, 1.55896277e+03, 5.44258423e+01,
        8.85968750e+02, 2.73610268e+01, 6.72265869e+02, 6.57043457e+01,
        5.45361145e+02, 3.21595840e+01, 4.21207977e+02, 5.85881538e+01,
        4.81918274e+02, 4.66307602e+01, 6.91869141e+02, 5.11259003e+01,
        5.74698853e+02, 3.61729660e+01, 5.71869446e+02, 3.47335777e+01,
        8.97628418e+02, 2.77397423e+01, 6.21096252e+02, 3.91444054e+01,
        6.83932556e+02, 3.40488434e+01, 5.29363342e+02, 3.69703217e+01,
        6.29729797e+02, 3.13654251e+01, 1.14323059e+03, 3.42121010e+01,
        9.10473206e+02])
    
    data_min = np.array([ 1.07107759e-01,  1.53447501e-02,  9.53487703e-04,  4.37953460e-08,
         4.72741636e+02,  8.11881305e+02,  4.99162910e+02,  1.18352033e+03,
         6.58336276e+02,  1.14510153e+03,  1.35253906e-02,  5.02260479e-06,
        -2.66721360e-02,  9.31230861e-23, -8.79393052e-03,  4.67204480e-08,
         2.43772111e+01, -6.62171631e+02,  2.51905384e+01, -1.20533924e+01,
         9.66593075e+00, -1.04249832e+02,  2.05522895e+00, -3.51384926e+01,
         3.54037285e+00, -4.78867798e+01,  9.75414085e+00, -3.48892632e+01,
         5.26781082e+00, -4.51870193e+01,  7.56150627e+00, -4.03234673e+01,
         6.89909983e+00, -3.94517517e+01,  8.25231361e+00, -3.28335457e+01,
         7.58491325e+00, -4.00081940e+01,  4.99889231e+00, -2.37591953e+01,
         2.34563255e+00, -2.93505001e+01,  7.80611753e+00, -2.33900909e+01,
         3.23007345e+00, -3.04670868e+01,  1.48191714e+00, -2.68500156e+01,
         1.32578599e+00, -2.78097954e+01,  1.62454367e+00, -2.07338085e+01,
         3.43743920e+00, -2.74484558e+01,  3.06530213e+00, -3.56406593e+01,
         2.82131255e-01])
    
    scale = np.array([1.55672788e+00, 9.46794517e+00, 2.26442474e+00, 3.06605714e+01,
       2.01621327e-04, 2.08624246e-07, 3.11625050e-04, 8.10399702e-07,
       1.13261697e-04, 7.70293853e-08, 2.99240210e+00, 1.53420456e+01,
       2.36070799e+01, 7.87082388e+00, 6.40503688e+01, 1.69839709e+01,
       3.80615751e-03, 1.29851083e-03, 2.22211435e-05, 3.79844124e-03,
       1.95223987e-04, 5.40260900e-03, 2.41211990e-04, 8.00923746e-03,
       4.34742616e-04, 1.05573226e-02, 6.45490854e-04, 1.11963144e-02,
       1.13545922e-03, 1.37839687e-02, 1.50442822e-03, 9.43148757e-03,
       1.85714111e-03, 1.39642696e-02, 2.42156747e-03, 1.09383221e-02,
       2.10822194e-03, 1.15421523e-02, 1.45587904e-03, 1.33537921e-02,
       1.74717284e-03, 1.52617079e-02, 1.77285058e-03, 1.72046951e-02,
       1.11807005e-03, 1.71801147e-02, 1.61390714e-03, 1.51527960e-02,
       1.46497229e-03, 1.61658908e-02, 1.89487679e-03, 1.73297821e-02,
       1.59669839e-03, 1.70027889e-02, 8.77065817e-04, 1.43158265e-02,
       1.09867041e-03])
    
    min_ = np.array([-1.66737634e-01, -1.45283253e-01, -2.15910114e-03, -1.34279033e-06,
       -9.53147961e-02, -1.69378125e-04, -1.55551667e-01, -9.59124521e-04,
       -7.45642834e-02, -8.82064670e-05, -4.04734074e-02, -7.70570317e-05,
        6.29651246e-01, -7.32955410e-22,  5.63254493e-01, -7.93498731e-07,
       -9.27835052e-02,  8.59837034e-01, -5.59762569e-04,  4.57841028e-02,
       -1.88702154e-03,  5.63221081e-01, -4.95745864e-04,  2.81432531e-01,
       -1.53915095e-03,  5.05556182e-01, -6.29620871e-03,  3.90631159e-01,
       -5.98138436e-03,  6.22856462e-01, -1.13757434e-02,  3.80310280e-01,
       -1.28126019e-02,  5.50914898e-01, -1.99835342e-02,  3.59143900e-01,
       -1.59906806e-02,  4.61780666e-01, -7.27778255e-03,  3.17275356e-01,
       -4.09822549e-03,  4.47938759e-01, -1.38390800e-02,  4.02419384e-01,
       -3.61144838e-03,  5.23428046e-01, -2.39167666e-03,  4.06852810e-01,
       -1.94223974e-03,  4.49570116e-01, -3.07831009e-03,  3.59312383e-01,
       -5.48855364e-03,  4.66700299e-01, -2.68847172e-03,  5.10225497e-01,
       -3.09969261e-04])
    
    X = df[df["filename"]==filename].drop(['filename', 'length', 'label'], axis=1)
    # Preprocessing
    scaler = MinMaxScaler()
    scaler.data_max_ = data_max
    scaler.data_min_ = data_min
    scaler.scale_ = scale
    scaler.min_ = min_
    X = pd.DataFrame(scaler.transform(X))
    return X

#uploading the file
upload= st.file_uploader('Provide a song', type=['wav'])
st.header('Input Audio')
c1, c2 = st.columns(2)

if upload is not None:
    
    audio_bytes = upload.read()

    # Converting bytes to audio array
    y, sr = librosa.load(BytesIO(audio_bytes), sr=None)
    
    # Generating the spectrogram
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

    
    c1.audio(audio_bytes)
    
    plt.style.use('default')

    # Displaying the waveplot
    fig= plt.figure(figsize=(10,5.5))
    plt.plot(y)
    plt.title('Waveplot')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    # plt.axis('off')
    c1.pyplot(fig)

    # Displaying the spectrogram
    fig2 = plt.figure(figsize=(10, 6.52))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    c2.pyplot(fig2)

    st.header("Output")
    #loading the model
    encoded_classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    model = tf.keras.saving.load_model('dense-gtzan.keras')
    pred = model.predict(audio_features(upload.name))
    st.write(f'Predicted Genre - {encoded_classes[np.argmax(pred)].capitalize()}')
    st.write(f'Confidence - {np.max(pred)*100:.2f}%')
    st.write("Note - The current accuracy of the model is approximately 98.40%, if you find the output to be incorrect, congratulations, you are in the top 1.6% :)")