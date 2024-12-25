import os
import uuid
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from skimage.io import imread
from skimage import color
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from skimage import measure
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib
from natsort import natsorted

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class ECG:
    def getImage(self, image):
        return imread(image)

    def GrayImgae(self, image):
        image_gray = color.rgb2gray(image)
        return resize(image_gray, (1572, 2213))

    def DividingLeads(self, image):
        leads = [
            image[300:600, 150:643],    # Lead 1
            image[300:600, 646:1135],   # Lead aVR
            image[300:600, 1140:1625],  # Lead V1
            image[300:600, 1630:2125],  # Lead V4
            image[600:900, 150:643],    # Lead 2
            image[600:900, 646:1135],   # Lead aVL
            image[600:900, 1140:1625],  # Lead V2
            image[600:900, 1630:2125],  # Lead V5
            image[900:1200, 150:643],   # Lead 3
            image[900:1200, 646:1135],  # Lead aVF
            image[900:1200, 1140:1625], # Lead V3
            image[900:1200, 1630:2125], # Lead V6
            image[1250:1480, 150:2125]  # Long Lead
        ]
        return leads

    def PreprocessingLeads(self, Leads):
        processed_leads = []
        for lead in Leads[:-1]:  # Process all except last lead
            grayscale = color.rgb2gray(lead)
            blurred_image = gaussian(grayscale, sigma=1)
            global_thresh = threshold_otsu(blurred_image)
            binary_global = blurred_image < global_thresh
            binary_global = resize(binary_global, (300, 450))
            processed_leads.append(binary_global)
        
        # Process last lead (Lead 13)
        grayscale = color.rgb2gray(Leads[-1])
        blurred_image = gaussian(grayscale, sigma=1)
        global_thresh = threshold_otsu(blurred_image)
        binary_global = blurred_image < global_thresh
        processed_leads.append(binary_global)
        
        return processed_leads

    def SignalExtraction_Scaling(self, Leads):
        for x, lead in enumerate(Leads[:-1]):
            grayscale = color.rgb2gray(lead)
            blurred_image = gaussian(grayscale, sigma=0.7)
            global_thresh = threshold_otsu(blurred_image)
            binary_global = blurred_image < global_thresh
            binary_global = resize(binary_global, (300, 450))
            
            contours = measure.find_contours(binary_global, 0.8)
            contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
            
            for contour in contours:
                if contour.shape in contours_shape:
                    test = resize(contour, (255, 2))
                    
            scaler = MinMaxScaler()
            fit_transform_data = scaler.fit_transform(test)
            Normalized_Scaled = pd.DataFrame(fit_transform_data[:,0], columns=['X'])
            Normalized_Scaled = Normalized_Scaled.T
            Normalized_Scaled.to_csv(f'Scaled_1DLead_{x+1}.csv', index=False)

    def CombineConvert1Dsignal(self):
        test_final = pd.read_csv('Scaled_1DLead_1.csv')
        location = os.getcwd()
        
        for file in natsorted(os.listdir(location)):
            if file.endswith(".csv") and file != 'Scaled_1DLead_1.csv':
                df = pd.read_csv(file)
                test_final = pd.concat([test_final, df], axis=1, ignore_index=True)
        
        return test_final

    def DimensionalReduciton(self, test_final):
        pca_loaded_model = joblib.load('PCA_ECG (1).pkl')
        result = pca_loaded_model.transform(test_final)
        return pd.DataFrame(result)

    def ModelLoad_predict(self, final_df):
        loaded_model = joblib.load('Heart_Disease_Prediction_using_ECG (4).pkl')
        result = loaded_model.predict(final_df)
        predictions = {
            0: "Abnormal Heartbeat",
            1: "Myocardial Infarction",
            2: "Normal",
            3: "History of Myocardial Infarction"
        }
        return predictions.get(result[0], "Unknown")

ecg = ECG()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/predict', methods=['POST'])
def prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            image = ecg.getImage(file_path)
            gray_image = ecg.GrayImgae(image)
            leads = ecg.DividingLeads(image)
            processed_leads = ecg.PreprocessingLeads(leads)
            ecg.SignalExtraction_Scaling(leads)
            signal_1d = ecg.CombineConvert1Dsignal()
            final_data = ecg.DimensionalReduciton(signal_1d)
            prediction_result = ecg.ModelLoad_predict(final_data)
            
            return jsonify({
                'id': str(uuid.uuid4()),
                'patientEmail': request.form.get('patientEmail', ''),
                'originalImageUrl': f"/uploads/{filename}",
                'processedImageUrl': f"/uploads/processed_{filename}",
                'prediction': prediction_result,
                'timestamp': str(uuid.uuid4())
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)