import tensorflow as tf

# ファイルパスを明示的に指定
input_file_path = r"C:\Users\syake\GitHub\NoticeEar_DCON2025\h5totflite\final_model.h5"
output_file_path = r"C:\Users\syake\GitHub\NoticeEar_DCON2025\h5totflite\final_model.tflite"

try:
    # .h5モデルをロード
    print(f"Loading model from: {input_file_path}")
    model = tf.keras.models.load_model(input_file_path)
    print("Model loaded successfully")

    # TensorFlow Lite形式に変換
    print("Converting model to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print("Model converted successfully")

    # .tfliteモデルを保存
    print(f"Saving TFLite model to: {output_file_path}")
    with open(output_file_path, 'wb') as f:
        f.write(tflite_model)
    print("Model saved successfully")

except FileNotFoundError:
    print(f"File not found: {input_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
