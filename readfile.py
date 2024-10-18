import os


def summarize_folder(folder_path, output_txt_path):
    with open(output_txt_path, 'w', encoding='utf-8') as output_file:
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_size = os.path.getsize(file_path)

                # 这里只提取文件名和大小作为示例
                # 对于文本文件，可以使用open()函数读取内容并提取摘要
                # 对于非文本文件，可以根据需要提取其他信息

                output_file.write(f"File Name: {file_name}\n")
                output_file.write(f"File Size: {file_size} bytes\n")
                output_file.write("\n")  # 添加空行作为文件之间的分隔

    print(f"Summary saved to {output_txt_path}")


# 示例用法
folder_path = "training-a"
output_txt_path = "datafile_explain.txt"
summarize_folder(folder_path, output_txt_path)