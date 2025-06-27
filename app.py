import streamlit as st
import numpy as np
import librosa
import os
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
import tempfile
import pandas as pd
import subprocess

# --- Các hằng số ---
CSV_FILE_PATH = "audio_features.csv"
DATASET_PATH = "percussion sound dataset/Gộp" 

# --- Các hàm xử lý ---

# Hàm trích xuất đặc trưng cho file input (tương tự như cũ)
def extract_features(file_path, sr=44100, n_mfcc=20):
    """Trích xuất và trả về cả đặc trưng thô và đặc trưng đã chuẩn hóa cho file input."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        if len(y) < 2048:
            st.warning("File âm thanh quá ngắn để xử lý.")
            return None, None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        flatness = librosa.feature.spectral_flatness(y=y)

        raw_features = {
            "MFCC (mean)": mfcc.mean(axis=1),
            "Zero-Crossing Rate (mean)": zcr.mean(),
            "Spectral Centroid (mean)": centroid.mean(),
            "Spectral Rolloff (mean)": rolloff.mean(),
            "RMS Energy (mean)": rms.mean(),
            "Spectral Flatness (mean)": flatness.mean()
        }

        feature_vector = np.hstack(list(raw_features.values()))
        scaler = StandardScaler()
        scaled_feature_vector = scaler.fit_transform(feature_vector.reshape(-1, 1)).flatten()
        
        return raw_features, scaled_feature_vector.astype(np.float64)
    except Exception as e:
        st.error(f"Lỗi khi xử lý file của bạn: {e}")
        return None, None

# Hàm tìm kiếm (không đổi)
def find_nearest_sounds(input_features, features_array, file_names, k=3):
    if input_features is None:
        return []
    
    kdtree = KDTree(features_array, metric='euclidean')
    distances, indices = kdtree.query(input_features.reshape(1, -1), k=k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "file_name": file_names[idx],
            "distance": distances[0][i]
        })
    return results

# Hàm tải đặc trưng từ CSV
@st.cache_data
def load_features_from_csv(csv_path):
    """Tải dữ liệu đặc trưng đã được tính toán sẵn từ file CSV."""
    df = pd.read_csv(csv_path)
    file_names = df['file_name'].tolist()
    feature_columns = df.columns.drop('file_name')
    features_array = df[feature_columns].to_numpy()
    return features_array, file_names

# --- Giao diện Streamlit ---

st.set_page_config(layout="wide")
st.title("Hệ thống truy xuất âm thanh bộ gõ")

# Kiểm tra sự tồn tại của file CSV
if not os.path.exists(CSV_FILE_PATH):
    st.warning(f"**File đặc trưng `{CSV_FILE_PATH}` không tồn tại.**")
    st.info("Để ứng dụng hoạt động, cần phải tính toán và lưu trữ đặc trưng của bộ dữ liệu vào file CSV.")
    
    if st.button("Bắt đầu tính toán đặc trưng (có thể mất vài phút)"):
        with st.spinner("Đang chạy script tiền xử lý... Vui lòng không đóng cửa sổ này."):
            try:
                # Tạo môi trường riêng cho tiến trình con để ép nó dùng UTF-8
                process_env = os.environ.copy()
                process_env["PYTHONIOENCODING"] = "utf-8"
                
                # Chạy script preprocess_data.py với môi trường đã thiết lập
                process = subprocess.run(
                    ["python", "preprocess_data.py"],
                    capture_output=True,
                    text=True,
                    check=True,
                    encoding='utf-8',
                    errors='replace',
                    env=process_env
                )
                st.success(f"Đã tạo file `{CSV_FILE_PATH}` thành công!")
                st.code(process.stdout)
                st.info("Vui lòng nhấn Rerun để bắt đầu sử dụng ứng dụng.")
            except FileNotFoundError:
                st.error("Lỗi: Lệnh 'python' không được tìm thấy. Hãy chắc chắn Python đã được cài đặt và thêm vào PATH.")
            except subprocess.CalledProcessError as e:
                st.error("Script tiền xử lý đã gặp lỗi:")
                st.code(e.stderr)
else:
    # Nếu file CSV tồn tại, chạy ứng dụng chính
    db_features, db_file_names = load_features_from_csv(CSV_FILE_PATH)
    st.success(f"Đã tải thành công đặc trưng từ `{CSV_FILE_PATH}` ({len(db_file_names)} file).")

    with st.expander("Xem danh sách các file trong dataset"):
        st.dataframe(pd.DataFrame({"File Name": db_file_names}))

    with st.expander("Xem nội dung file CSV đã xử lý (audio_features.csv)"):
        df = pd.read_csv(CSV_FILE_PATH)
        st.dataframe(df)

    st.header("Bước 1: Tải lên file âm thanh của bạn")
    uploaded_file = st.file_uploader("Tải lên một file âm thanh (WAV, MP3) để tìm kiếm", type=["wav", "mp3"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.subheader("Âm thanh đã tải lên")
        st.audio(tmp_file_path)

        st.header("Bước 2: Phân tích và Trích xuất Đặc trưng")
        with st.spinner("Đang trích xuất đặc trưng từ file của bạn..."):
            raw_features, input_features = extract_features(tmp_file_path)
        
        os.remove(tmp_file_path)

        if raw_features and input_features is not None:
            st.info("Quá trình trích xuất đặc trưng hoàn tất.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Các đặc trưng thô")
                prepared_features = []
                for key, value in raw_features.items():
                    if isinstance(value, np.ndarray):
                        for i, v in enumerate(value):
                            prepared_features.append((f"{key} [{i}]", v))
                    else:
                        prepared_features.append((key, value))
                
                features_df = pd.DataFrame.from_records(prepared_features, columns=['Đặc trưng', 'Giá trị'])
                st.dataframe(features_df)

            with col2:
                st.subheader("Vector đặc trưng cuối cùng (đã chuẩn hóa)")
                st.write(input_features)

            st.header("Bước 3: Tìm kiếm")
            if st.button("Tìm kiếm âm thanh tương tự"):
                with st.spinner("Đang tìm kiếm..."):
                    st.write(f"Đang so sánh vector đặc trưng của bạn với {len(db_file_names)} vector trong dataset...")
                    st.write("Sử dụng thuật toán `KDTree` với metric `euclidean` để tìm 3 kết quả gần nhất.")
                    
                    with st.expander("Xem công thức tính độ tương đồng"):
                        st.info("Hệ thống sử dụng **khoảng cách Euclidean** để đo độ tương đồng. Khoảng cách càng nhỏ, hai âm thanh càng giống nhau.")
                        st.latex(r'''
                        d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \cdots + (p_n - q_n)^2} = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
                        ''')
                        st.markdown("""
                        Trong đó:
                        - **p**: là vector đặc trưng của file bạn tải lên.
                        - **q**: là vector đặc trưng của một file trong dataset.
                        - **n**: là số chiều của vector đặc trưng (ở đây là 25).
                        """)

                    results = find_nearest_sounds(input_features, db_features, db_file_names, k=3)
                
                st.header("Kết quả tìm kiếm (Top 3)")

                if not results:
                    st.warning("Không tìm thấy kết quả nào.")
                else:
                    for result in results:
                        st.subheader(f"File: {result['file_name']}")
                        st.write(f"Độ tương đồng (distance): {result['distance']:.4f} (càng nhỏ càng giống)")
                        result_file_path = os.path.join(DATASET_PATH, result['file_name'])
                        st.audio(result_file_path)
                        st.divider() 