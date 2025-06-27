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
DATASET_PATH = "percussion sound dataset/Gop"
N_MFCC = 20
SR = 44100
SEGMENT_DURATION = 1.0
OVERLAP_RATIO = 0.5

# --- Các hàm xử lý ---

def extract_segment_features(file_path):
    """Trích xuất đặc trưng từ các đoạn của file âm thanh tải lên."""
    try:
        y, sr_loaded = librosa.load(file_path, sr=SR)
        if len(y) < int(SEGMENT_DURATION * SR):
            st.warning("File âm thanh quá ngắn để chia đoạn.")
            return None
            
        segment_samples = int(SEGMENT_DURATION * SR)
        hop_length = int(segment_samples * (1 - OVERLAP_RATIO))
        segments = [y[i:i + segment_samples] for i in range(0, len(y) - segment_samples + 1, hop_length)
                    if len(y[i:i + segment_samples]) == segment_samples]

        all_features = []
        for segment in segments:
            mfcc = librosa.feature.mfcc(y=segment, sr=SR, n_mfcc=N_MFCC)
            zcr = librosa.feature.zero_crossing_rate(segment)
            centroid = librosa.feature.spectral_centroid(y=segment, sr=SR)
            rolloff = librosa.feature.spectral_rolloff(y=segment, sr=SR)
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
        st.error(f"Lỗi khi xử lý file của bạn: {e}")
        return None

def find_nearest_sounds_segmented(input_segments_features, db_features, db_mapping, k=5):
    """Tìm kiếm file tương tự dựa trên các đoạn tương đồng nhất."""
    if input_segments_features is None or len(input_segments_features) == 0:
        return []

    # Xây dựng KD-Tree từ tất cả các đoạn trong dataset
    kdtree = KDTree(db_features, metric='euclidean')

    file_scores = {}
    # Với mỗi đoạn của file input, tìm các đoạn gần nhất trong dataset
    for input_feature in input_segments_features:
        distances, indices = kdtree.query([input_feature], k=k)
        for i, idx in enumerate(indices[0]):
            # Lấy thông tin file và segment từ mapping
            mapping_info = db_mapping[idx]
            file_name = mapping_info["file_name"]
            distance = float(distances[0][i])
            
            # Nếu file này chưa có trong score, hoặc distance này nhỏ hơn, cập nhật nó
            if file_name not in file_scores or distance < file_scores[file_name]:
                file_scores[file_name] = distance

    # Chuyển dict thành list và sắp xếp
    results = [{"file_name": name, "min_distance": score} for name, score in file_scores.items()]
    results.sort(key=lambda x: x['min_distance'])
    
    return results[:k]

@st.cache_data
def load_feature_df(csv_path):
    """Tải toàn bộ dữ liệu đặc trưng dưới dạng một pandas DataFrame."""
    return pd.read_csv(csv_path)

def get_features_and_mapping_from_df(df):
    """Trích xuất mảng đặc trưng và mapping từ DataFrame."""
    db_mapping = df[['file_name', 'segment_index']].to_dict('records')
    feature_columns = [col for col in df.columns if col not in ['file_name', 'segment_index']]
    features_array = df[feature_columns].to_numpy(dtype=np.float64)
    return features_array, db_mapping

# --- Giao diện Streamlit ---

st.set_page_config(layout="wide")
st.title("BTL HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN - HỆ CSDL LƯU TRỮ VÀ TÌM KIẾM NHẠC CỤ THUỘC BỘ GÕ")

if not os.path.exists(CSV_FILE_PATH):
    # Giao diện khi chưa có file CSV (không đổi)
    st.warning(f"**File đặc trưng `{CSV_FILE_PATH}` không tồn tại.**")
    st.info("Để ứng dụng hoạt động, cần phải tính toán và lưu trữ đặc trưng của bộ dữ liệu vào file CSV.")
    if st.button("Bắt đầu tính toán đặc trưng (có thể mất vài phút)"):
        with st.spinner("Đang chạy script tiền xử lý... Vui lòng không đóng cửa sổ này."):
            try:
                process_env = os.environ.copy()
                process_env["PYTHONIOENCODING"] = "utf-8"
                process = subprocess.run(
                    ["python", "preprocess_data.py"],
                    capture_output=True, text=True, check=True,
                    encoding='utf-8', errors='replace', env=process_env
                )
                st.success(f"Đã tạo file `{CSV_FILE_PATH}` thành công!")
                st.code(process.stdout)
                st.info("Vui lòng nhấn Rerun để bắt đầu sử dụng ứng dụng.")
            except FileNotFoundError:
                 st.error("Lỗi: Lệnh 'python' không được tìm thấy.")
            except subprocess.CalledProcessError as e:
                st.error("Script tiền xử lý đã gặp lỗi:")
                st.code(e.stderr)
else:
    # Giao diện chính khi đã có file CSV
    feature_df = load_feature_df(CSV_FILE_PATH)
    db_features, db_mapping = get_features_and_mapping_from_df(feature_df)
    st.success(f"Đã tải thành công đặc trưng của {len(db_features)} đoạn từ `{CSV_FILE_PATH}`.")

    with st.expander("Xem nội dung file CSV đã xử lý (audio_features.csv)"):
        st.dataframe(feature_df)

    st.header("Bước 1: Tải lên file âm thanh của bạn")
    uploaded_file = st.file_uploader("Tải lên file âm thanh (WAV, MP3) để tìm kiếm", type=["wav", "mp3"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.subheader("Âm thanh đã tải lên")
        st.audio(tmp_file_path)

        st.header("Bước 2: Phân tích và Trích xuất Đặc trưng theo từng đoạn")

        with st.expander("Giải thích các đặc trưng âm thanh được sử dụng"):
            st.markdown("""
            Hệ thống trích xuất các đặc trưng sau từ mỗi đoạn âm thanh dài 1 giây để tạo ra một "vector đặc trưng" đại diện cho đoạn đó.
            Vector này sau đó được chuẩn hóa để đảm bảo các đặc trưng có thang đo khác nhau không ảnh hưởng đến kết quả.
            """)
            
            st.subheader("1. MFCC (Mel-Frequency Cepstral Coefficients)")
            st.markdown("Là một tập hợp các hệ số mô tả hình dạng tổng thể của phổ âm thanh. Đây là một trong những đặc trưng quan trọng nhất để nhận dạng âm thanh và giọng nói. Chúng ta lấy giá trị trung bình của 20 hệ số MFCC đầu tiên.")

            st.subheader("2. Zero-Crossing Rate (Tỷ lệ băng qua không)")
            st.markdown("Đo tốc độ thay đổi của tín hiệu bằng cách đếm số lần tín hiệu đi qua giá trị 0. Đặc trưng này hữu ích để phân biệt âm thanh có âm vực và âm thanh nhiễu (âm thanh gõ so với âm thanh xì).")
            st.latex(r''' ZCR = \frac{1}{T-1} \sum_{t=1}^{T-1} \mathbb{1}(s_t \cdot s_{t-1} < 0) ''')
            st.markdown(r"Trong đó $s$ là tín hiệu và $\mathbb{1}$ là hàm chỉ thị (indicator function).")

            st.subheader("3. Spectral Centroid (Trọng tâm phổ)")
            st.markdown("Xác định 'trung tâm khối lượng' của phổ, cho biết tần số nào chiếm ưu thế. Giá trị cao hơn tương ứng với âm thanh 'sáng' hơn.")
            st.latex(r''' C_t = \frac{\sum_{n=1}^{N} f(n) x_t(n)}{\sum_{n=1}^{N} x_t(n)} ''')
            st.markdown(r"Trong đó $x_t(n)$ là biên độ của tần số $f(n)$ tại frame $t$.")

            st.subheader("4. Spectral Rolloff (Ngưỡng cuốn phổ)")
            st.markdown("Là tần số mà dưới nó chứa một tỷ lệ phần trăm xác định (thường là 85%) của tổng năng lượng phổ. Nó đo lường sự lệch của hình dạng phổ.")

            st.subheader("5. RMS Energy (Năng lượng RMS)")
            st.markdown("Đo lường biên độ (độ to) của tín hiệu. Đây là căn bậc hai của trung bình bình phương của tín hiệu trong một khoảng thời gian.")
            st.latex(r''' RMS = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} [x(n)]^2} ''')
            st.markdown(r"Trong đó $x(n)$ là giá trị của mẫu tín hiệu.")
            
            st.subheader("6. Spectral Flatness (Độ phẳng phổ)")
            st.markdown("Đo lường mức độ 'giống nhiễu' của một phổ. Giá trị thấp cho thấy các đỉnh phổ rõ rệt (âm thanh có âm vực), trong khi giá trị cao cho thấy phổ phẳng hơn, giống nhiễu trắng. Nó là tỉ lệ giữa trung bình nhân và trung bình cộng của phổ năng lượng.")
            st.latex(r''' \text{Flatness} = \frac{ \left( \prod_{n=0}^{N-1} p(n) \right)^{\frac{1}{N}} }{ \frac{1}{N} \sum_{n=0}^{N-1} p(n) } ''')
            st.markdown(r"Trong đó $p(n)$ là năng lượng (power) của tần số thứ $n$.")
            
        input_segments_features = extract_segment_features(tmp_file_path)
        os.remove(tmp_file_path)

        if input_segments_features is not None:
            st.info(f"Quá trình trích xuất hoàn tất. File của bạn được chia thành {len(input_segments_features)} đoạn để phân tích.")
            with st.expander("Xem các vector đặc trưng của file tải lên"):
                st.write(input_segments_features)

            st.header("Bước 3: Tìm kiếm")
            if st.button("Tìm kiếm âm thanh tương tự"):
                with st.spinner("Đang tìm kiếm..."):
                    results = find_nearest_sounds_segmented(input_segments_features, db_features, db_mapping, k=5)
                
                st.header("Kết quả tìm kiếm (Top 5)")

                with st.expander("Giải thích về 'min_distance' (Khoảng cách Euclidean)"):
                    st.markdown("Khoảng cách được sử dụng để đo lường sự khác biệt giữa hai vector đặc trưng (vector của một đoạn trong file bạn tải lên và vector của một đoạn trong cơ sở dữ liệu). Chúng tôi sử dụng **Khoảng cách Euclidean**.")
                    st.latex(r''' D(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2} ''')
                    st.markdown(r"Trong đó $p = (p_1, ..., p_n)$ và $q = (q_1, ..., q_n)$ là hai vector đặc trưng. Giá trị khoảng cách càng nhỏ, hai đoạn âm thanh càng giống nhau.")

                if not results:
                    st.warning("Không tìm thấy kết quả nào.")
                else:
                    for result in results:
                        st.subheader(f"File: {result['file_name']}")
                        st.write(f"Độ tương đồng tốt nhất (min_distance): {result['min_distance']} (càng nhỏ càng giống)")
                        result_file_path = os.path.join(DATASET_PATH, result['file_name'])
                        st.audio(result_file_path)
                        st.divider() 