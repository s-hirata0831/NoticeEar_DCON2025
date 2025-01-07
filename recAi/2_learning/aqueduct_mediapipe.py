import mediapipe as mp
import tensorflow as tf
import numpy as np
import librosa  # 音声ファイルの読み込みに使用

# MediaPipe AudioClassifierの初期化
mp_audio = mp.tasks.audio # mp_audio を定義
audio_classifier_options = mp_audio.AudioClassifierOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/2_learning/yamnet.tflite'),
    max_results=521,
    score_threshold=0.001
)
audio_classifier = mp_audio.AudioClassifier.create_from_options(audio_classifier_options)

# 自作モデルの読み込みとTensorFlow Liteへの変換
try:
    custom_model = tf.keras.models.load_model('/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/2_learning/final_model.h5')  # 自作モデルのパス
except OSError as e:
    print(f"Error loading model: {e}")
    exit()

converter = tf.lite.TFLiteConverter.from_keras_model(custom_model)
tflite_model = converter.convert()
with open('custom_model.tflite', 'wb') as f:
    f.write(tflite_model)

# TensorFlow Liteインタプリタの初期化
interpreter = tf.lite.Interpreter(model_path="custom_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def inference(audio_data, sample_rate):
    # 音声データをfloat32に変換 (重要)
    audio_data = audio_data.astype(np.float32)

    # AudioTensorを作成
    audio_tensor = mp_audio.AudioTensor.create(audio_data.tobytes(), sample_rate, 1)

    # YAMNetで推論
    classification_result = audio_classifier.classify(audio_tensor)
    if classification_result.classifications:
        yamnet_output = np.array([c.score for c in classification_result.classifications[0].categories])
    else:
        print("No YAMNet classifications found.")
        return None

    # 自作モデルの入力データを作成
    input_data = np.expand_dims(yamnet_output, axis=0).astype(np.float32)

    # 自作モデルで推論
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    custom_model_output = interpreter.get_tensor(output_details[0]['index'])

    return custom_model_output


# 音声ファイルの読み込みと推論の実行例
if __name__ == "__main__":
    try:
        audio_data, sample_rate = librosa.load('/Users/hiratasoma/Documents/NoticeEar_DCON2025/recAi/2_learning/siren.mp3', sr=None)  # 音声ファイルのパス
    except FileNotFoundError:
        print("Audio file not found.")
        exit()
    except Exception as e:
        print(f"Error loading audio: {e}")
        exit()

    custom_model_result = inference(audio_data, sample_rate)

    if custom_model_result is not None:
        print("Custom Model Output:", custom_model_result)
        # 結果の解釈 (例: 最も確率の高いクラスのインデックスを取得)
        predicted_class = np.argmax(custom_model_result)
        print("Predicted Class:", predicted_class)
        classes = ["horn", "microwave", "animals", "guns", "siren"]
        print(f"Predicted Class Name: {classes[predicted_class]}")
