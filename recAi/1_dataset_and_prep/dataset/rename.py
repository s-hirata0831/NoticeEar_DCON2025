import os

def remove_leading_hyphen_and_before(directory):
    """
    指定されたディレクトリ内のファイルから、先頭のハイフンとそれ以前の文字を削除します。

    Args:
        directory: 対象のディレクトリのパス

    Returns:
        None
    """

    for filename in os.listdir(directory):
        # ファイルのフルパスを取得
        file_path = os.path.join(directory, filename)

        # ファイルがファイルであれば処理
        if os.path.isfile(file_path):
            # 先頭のハイフンとそれ以前の文字を削除した新しいファイル名を作成
            new_filename = filename.split('-', 1)[1] if '-' in filename else filename
            new_file_path = os.path.join(directory, new_filename)

            # ファイルを移動
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

# 処理するディレクトリを指定
target_directory = "/Users/hiratasoma/Desktop/DCON/UrbanSound8K/audio/fold10"

# 関数を呼び出す
remove_leading_hyphen_and_before(target_directory)