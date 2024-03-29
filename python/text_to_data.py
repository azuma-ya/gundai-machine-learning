import sys
import re
import os

path = "./text/"

files = os.listdir(path)

for file_name in files:
    file_path = path + file_name
    # print(file_path)
    with open(file_path, "r", encoding="Shift_JIS") as file:
        source_text = file.read()
        # 本文前の注釈にタグを埋め込んで、そこを元に本文を抽出
        text_tagging_hi = re.sub(r"--+", "タグを埋め込みます", source_text)
        text_remove_tag = text_tagging_hi.split("タグを埋め込みます")[-1]
        # 単語に振ってあるルビを削除
        text_without_rubi = re.sub(r"《.+?》", "", text_remove_tag)
        # 本文中にある注釈や解説を削除
        text_without_com = re.sub(r"［.+?］", "", text_without_rubi)
        # 出版社や作成日などの情報を削除
        output = text_without_com.split("底本")[0]
        output_file = open(
            f"./text_data/preprocess_done_{file_name}", "a", encoding="utf-8"
        ).write(output)
