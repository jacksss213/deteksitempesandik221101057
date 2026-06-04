#!/usr/bin/env python3
"""
Server API Deteksi Kualitas Tempe
Sandik Bayu Asmoro (221101057)
Cara pakai:  python3 server.py
API: POST /predict  |  GET /model-info  |  GET /health
"""
import os, io, time, base64, warnings
import numpy as np
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, make_response
import joblib, cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

app = Flask(__name__)

# ── Manual CORS ────────────────────────────────────────────────
@app.after_request
def add_cors(resp):
    resp.headers['Access-Control-Allow-Origin']  = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp

@app.route('/predict',    methods=['OPTIONS'])
@app.route('/model-info', methods=['OPTIONS'])
@app.route('/health',     methods=['OPTIONS'])
def preflight():
    r = make_response()
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return r

# ── Load Models ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rf_model = dt_model = None

def load_models():
    global rf_model, dt_model
    try:
        rf_model = joblib.load(os.path.join(BASE_DIR, "random_forest_tempe_model.pkl"))
        print(f"✅ RF loaded | features={rf_model.n_features_in_} | trees={rf_model.n_estimators}")
    except Exception as e:
        print(f"⚠️  RF gagal: {e}")
    try:
        dt_model = joblib.load(os.path.join(BASE_DIR, "decision_tree_tempe_model.pkl"))
        print(f"✅ DT loaded | features={dt_model.n_features_in_}")
    except Exception as e:
        print(f"⚠️  DT gagal: {e}")

load_models()

# ── Constants ──────────────────────────────────────────────────
FEATURE_NAMES = (
    ['mean_R','mean_G','mean_B','std_R','std_G','std_B',
     'mean_H','mean_S','mean_V','std_H','std_S','std_V'] +
    [f'hist_R_{i}' for i in range(16)] +
    [f'hist_G_{i}' for i in range(16)] +
    [f'hist_B_{i}' for i in range(16)] +
    [f'GLCM_{p}_{a}' for p in ['contrast','correlation','energy','homogeneity','dissimilarity']
                     for a in ['0°','45°','90°','135°']] +
    [f'LBP_{i}' for i in range(26)]
)
LABELS = {0:'baik', 1:'sedang', 2:'buruk'}
DISPLAY = {
    'baik':   'Tempe Berkualitas Baik',
    'sedang': 'Tempe Berkualitas Sedang',
    'buruk':  'Tempe Berkualitas Buruk',
}
DESC = {
    'baik':   'Tempe berwarna putih cerah dengan pertumbuhan jamur merata, tekstur padat, dan aroma khas tempe segar yang baik.',
    'sedang': 'Tempe mengalami perubahan warna dan tekstur. Kualitas mulai menurun, masih bisa dikonsumsi namun perlu diperhatikan.',
    'buruk':  'Tempe gelap dengan indikasi pembusukan, tekstur rusak, dan warna tidak normal. Tidak disarankan untuk dikonsumsi.',
}

# ── Feature Extraction ─────────────────────────────────────────
def extract_color_features(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hr = cv2.calcHist([rgb],[0],None,[16],[0,256]).flatten()
    hg = cv2.calcHist([rgb],[1],None,[16],[0,256]).flatten()
    hb = cv2.calcHist([rgb],[2],None,[16],[0,256]).flatten()
    return np.concatenate([
        np.mean(rgb,axis=(0,1)), np.std(rgb,axis=(0,1)),
        np.mean(hsv,axis=(0,1)), np.std(hsv,axis=(0,1)),
        hr, hg, hb
    ])

def extract_glcm_features(gray):
    glcm = graycomatrix(gray, distances=[1,2,3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    feats = []
    for p in ['contrast','correlation','energy','homogeneity','dissimilarity']:
        feats.extend(graycoprops(glcm, p)[0])
    return np.array(feats)

def extract_lbp_features(gray):
    n_pts = 24
    lbp = local_binary_pattern(gray, n_pts, 3, method='uniform')
    h, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_pts+3), range=(0, n_pts+2))
    h = h.astype('float')
    h /= (h.sum() + 1e-6)
    return h

def extract_features(bgr):
    img  = cv2.resize(bgr, (128,128))
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3,3), 0)
    return np.concatenate([extract_color_features(img),
                           extract_glcm_features(gray),
                           extract_lbp_features(gray)])

def decode_img(raw_bytes):
    arr = np.frombuffer(raw_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ── Endpoints ──────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','rf_loaded':rf_model is not None,'dt_loaded':dt_model is not None})

@app.route('/model-info', methods=['GET'])
def model_info():
    rf_info = dt_info = None
    if rf_model is not None:
        imp = rf_model.feature_importances_
        top = np.argsort(imp)[-10:][::-1]
        rf_info = {
            'n_estimators': rf_model.n_estimators,
            'n_features':   int(rf_model.n_features_in_),
            'top10': [{'name': FEATURE_NAMES[i], 'importance': round(float(imp[i])*100,2)} for i in top]
        }
    if dt_model is not None:
        dt_info = {'n_features': int(dt_model.n_features_in_)}
    return jsonify({'status':'ok','random_forest':rf_info,'decision_tree':dt_info})

@app.route('/predict', methods=['POST'])
def predict():
    if rf_model is None:
        return jsonify({'error':'Model RF belum dimuat.'}), 503

    img_bgr = None
    try:
        if 'image' in request.files:
            img_bgr = decode_img(request.files['image'].read())
        elif request.is_json:
            b64 = request.get_json().get('image_base64','')
            if ',' in b64: b64 = b64.split(',',1)[1]
            img_bgr = decode_img(base64.b64decode(b64))
        else:
            return jsonify({'error':'Kirim via multipart "image" atau JSON "image_base64"'}), 400
    except Exception as e:
        return jsonify({'error': f'Gagal decode gambar: {e}'}), 400

    if img_bgr is None:
        return jsonify({'error':'Gambar tidak valid.'}), 400

    t0 = time.time()
    try:
        feats = extract_features(img_bgr).reshape(1,-1)
    except Exception as e:
        return jsonify({'error': f'Ekstraksi fitur gagal: {e}'}), 500

    rf_pred  = int(rf_model.predict(feats)[0])
    rf_proba = rf_model.predict_proba(feats)[0].tolist()

    dt_pred = dt_proba = None
    if dt_model:
        try:
            dt_pred  = int(dt_model.predict(feats)[0])
            dt_proba = dt_model.predict_proba(feats)[0].tolist()
        except: pass

    elapsed = round((time.time()-t0)*1000)
    imp = rf_model.feature_importances_
    top10 = [{'name': FEATURE_NAMES[i], 'importance': round(float(imp[i])*100,2)}
             for i in np.argsort(imp)[-10:][::-1]]

    label = LABELS[rf_pred]
    f = feats[0]
    return jsonify({
        'status':        'ok',
        'label_key':     label,
        'label_display': DISPLAY[label],
        'description':   DESC[label],
        'confidence':    round(max(rf_proba)*100, 1),
        'probabilities': {'baik':round(rf_proba[0]*100,1),'sedang':round(rf_proba[1]*100,1),'buruk':round(rf_proba[2]*100,1)},
        'decision_tree': {
            'label_key': LABELS.get(dt_pred) if dt_pred is not None else None,
            'probabilities': {'baik':round(dt_proba[0]*100,1),'sedang':round(dt_proba[1]*100,1),'buruk':round(dt_proba[2]*100,1)} if dt_proba else None,
        },
        'n_estimators':  rf_model.n_estimators,
        'n_features':    int(rf_model.n_features_in_),
        'elapsed_ms':    elapsed,
        'feature_values': {
            'mean_R': round(float(f[0]),1), 'mean_G': round(float(f[1]),1), 'mean_B': round(float(f[2]),1),
            'mean_H': round(float(f[6]),1), 'mean_S': round(float(f[7]),1), 'mean_V': round(float(f[8]),1),
            'glcm_contrast': round(float(f[60]),3), 'glcm_energy': round(float(f[62]),4),
            'glcm_homogeneity': round(float(f[63]),4),
        },
        'top10_features': top10,
    })

if __name__ == '__main__':
    print("="*55)
    print("  Deteksi Kualitas Tempe API - port 5000")
    print("  Buka file HTML di browser setelah server jalan")
    print("="*55)
    app.run(host='0.0.0.0', port=5000, debug=False)
