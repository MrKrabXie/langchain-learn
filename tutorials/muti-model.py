import base64
from io import BytesIO
from IPython.display import HTML, display
from PIL import Image
from langchain_ollama import OllamaLLM

# 将 PIL 图像转换为 Base64 字符串
def convert_to_base64(pil_image):
    """
    将 PIL 图像转换为 Base64 编码的字符串

    :param pil_image: PIL 图像
    :return: Base64 字符串
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # 可以根据需要更改格式
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# 使用 Base64 字符串显示图像
def plt_img_base64(img_base64):
    """
    通过 Base64 编码的字符串显示图像

    :param img_base64: Base64 字符串
    """
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    display(HTML(image_html))

# 加载并将图像转换为 Base64
file_path = "../img/cat.jpg"  # 根据需要调整路径
pil_image = Image.open(file_path)
image_b64 = convert_to_base64(pil_image)

# 显示图像（可选）
plt_img_base64(image_b64)

# 使用 'bakllava' 模型初始化 Ollama LLM
llm = OllamaLLM(model="bakllava")

# 将图像上下文绑定到 LLM
llm_with_image_context = llm.bind(images=[image_b64])

# 使用图像上下文向 LLM 提出一个商业问题
response = llm_with_image_context.invoke("what's this??")
print(response)

# 预期输出（仅为示例）
# '90%'
