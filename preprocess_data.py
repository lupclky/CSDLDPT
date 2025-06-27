import numpy as np
import librosa
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys

# --- Các hằng số và cấu hình ---
DATASET_PATH = "percussion sound dataset/Gop"
CSV_FILE_PATH = "audio_features.csv"
N_MFCC = 20
SR = 44100
SEGMENT_DURATION = 1.0  # Độ dài mỗi đoạn (giây)
OVERLAP_RATIO = 0.5     # Tỷ lệ chồng chéo giữa các đoạn

def extract_segment_features(file_path, sr=SR, n_mfcc=N_MFCC, segment_duration=SEGMENT_DURATION, overlap_ratio=OVERLAP_RATIO):
    """
    Trích xuất đặc trưng từ các đoạn (segment) của file âm thanh.
    Trả về một danh sách các vector đặc trưng, mỗi vector cho một đoạn.
    """
    try:
        y, sr_loaded = librosa.load(file_path, sr=sr)
        
        segment_samples = int(segment_duration * sr)
        if len(y) < segment_samples:
            return None # File quá ngắn
            
        hop_length = int(segment_samples * (1 - overlap_ratio))
        segments = [y[i:i + segment_samples] for i in range(0, len(y) - segment_samples + 1, hop_length)
                    if len(y[i:i + segment_samples]) == segment_samples]

        all_features = []
        for segment in segments:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
            zcr = librosa.feature.zero_crossing_rate(segment)
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
            rms = librosa.feature.rms(y=segment)
            flatness = librosa.feature.spectral_flatness(y=segment)

            feature_vector = np.hstack([
                mfcc.mean(axis=1), zcr.mean(), centroid.mean(),
                rolloff.mean(), rms.mean(), flatness.mean()
            ])

            scaler = StandardScaler()
            feature_vector = scaler.fit_transform(feature_vector.reshape(-1, 1)).flatten()
            all_features.append(feature_vector.astype(np.float64))

        return np.array(all_features) if all_features else None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_feature_csv(dataset_path, csv_path):
    """
    Quét qua dataset, trích xuất đặc trưng cho từng đoạn và lưu vào CSV.
    """
    print("Bắt đầu quá trình trích xuất đặc trưng theo từng đoạn...")
    features_data = []
    
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
        segment_features = extract_segment_features(file_path)
        
        if segment_features is not None:
            for seg_index, feature_vector in enumerate(segment_features):
                # Thêm tên file và chỉ số đoạn vào đầu
                row = [file_name, seg_index] + feature_vector.tolist()
                features_data.append(row)
        
        print(f"Đã xử lý {i+1}/{total_files}: {file_name} ({len(segment_features) if segment_features is not None else 0} đoạn)")

    # Tạo tên cột
    feature_names = [f'feature_{i}' for i in range(N_MFCC + 5)]
    columns = ['file_name', 'segment_index'] + feature_names
    
    # Tạo và lưu DataFrame
    df = pd.DataFrame.from_records(features_data, columns=columns)
    df.to_csv(csv_path, index=False)
    print(f"\nHoàn tất! Đã lưu thành công {len(df)} vector đặc trưng vào file '{csv_path}'")

if __name__ == "__main__":
    create_feature_csv(DATASET_PATH, CSV_FILE_PATH) 