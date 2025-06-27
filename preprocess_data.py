import numpy as np
import librosa
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys

# --- Các hằng số và cấu hình ---
DATASET_PATH = "percussion sound dataset/Gộp"
CSV_FILE_PATH = "audio_features.csv"
N_MFCC = 20
SR = 44100

def extract_features_for_csv(file_path, sr=SR, n_mfcc=N_MFCC):
    """Trích xuất vector đặc trưng đã chuẩn hóa cho việc lưu trữ."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        
        # Tránh lỗi cho các file quá ngắn
        if len(y) < 2048: # Cần đủ sample cho ít nhất một frame
            print(f"Warning: File is too short to process, skipping: {file_path}")
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        flatness = librosa.feature.spectral_flatness(y=y)

        feature_vector = np.hstack([
            mfcc.mean(axis=1),
            zcr.mean(),
            centroid.mean(),
            rolloff.mean(),
            rms.mean(),
            flatness.mean()
        ])

        scaler = StandardScaler()
        scaled_feature_vector = scaler.fit_transform(feature_vector.reshape(-1, 1)).flatten()
        
        return scaled_feature_vector.astype(np.float64)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_feature_csv(dataset_path, csv_path):
    """
    Quét qua thư mục dataset, trích xuất đặc trưng và lưu vào file CSV.
    """
    print("Bắt đầu quá trình trích xuất đặc trưng...")
    features_data = []
    
    # Lấy danh sách file audio
    try:
        audio_files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith(('.wav', '.mp3'))])
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục dataset tại '{dataset_path}'")
        sys.exit(1)
        
    if not audio_files:
        print("Không tìm thấy file audio nào trong thư mục dataset.")
        return

    total_files = len(audio_files)
    for i, file_name in enumerate(audio_files):
        file_path = os.path.join(dataset_path, file_name)
        feature_vector = extract_features_for_csv(file_path)
        
        if feature_vector is not None:
            # Thêm tên file vào đầu vector đặc trưng
            row = [file_name] + feature_vector.tolist()
            features_data.append(row)
        
        # In tiến trình
        print(f"Đã xử lý {i+1}/{total_files}: {file_name}")

    # Tạo tên cột
    feature_names = [f'feature_{i}' for i in range(N_MFCC + 5)]
    columns = ['file_name'] + feature_names
    
    # Tạo và lưu DataFrame
    df = pd.DataFrame.from_records(features_data, columns=columns)
    df.to_csv(csv_path, index=False)
    print(f"\nHoàn tất! Đã lưu thành công đặc trưng vào file '{csv_path}'")

if __name__ == "__main__":
    create_feature_csv(DATASET_PATH, CSV_FILE_PATH) 