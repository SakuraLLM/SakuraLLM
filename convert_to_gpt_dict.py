import json

# 这个脚本可以将轻小说机翻机器人网站里导出的术语表转化为模型适配的术语表。
# 输入一个文件src.json，内容是导出的文本。
# 输出的文件的路径是translate_epub.py的参数--gpt_dict_path的取值。

with open("src.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

with open("gpt_dict.txt", 'w', encoding='utf-8') as f:
    for key in data.keys():
        f.write(f"{key}->{data[key]}\n")
print("Model compatible gpt dict file: gpt_dict.txt generated.")