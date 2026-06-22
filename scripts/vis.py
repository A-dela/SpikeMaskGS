import numpy as np
from PIL import Image

def invert_mask(input_path, output_path):
    """
    读取黑白mask图像，将0和1值互换，然后保存结果
    
    参数:
        input_path: 输入mask图像的路径
        output_path: 输出处理后图像的路径
    """
    try:
        # 打开图像并转换为灰度模式
        img = Image.open(input_path).convert('L')
        
        # 将图像转换为numpy数组
        img_array = np.array(img)
        
        # 确保数组只包含0和1（如果有其他值，会被视为1处理）
        # 将非0值设为1
        img_array = (img_array > 0).astype(np.uint8)
        
        # 反转0和1（0变成1，1变成0）
        inverted_array = 1 - img_array
        
        # 将数组转换回图像
        inverted_img = Image.fromarray(inverted_array * 255)  # 乘以255是因为图像通常用0-255表示
        
        # 保存处理后的图像
        inverted_img.save(output_path)
        print(f"处理完成，结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"处理图像时出错: {e}")

if __name__ == "__main__":
    # 在这里指定输入和输出路径
    input_image_path = "data/vis/05699.png"   # 输入mask图像的路径
    output_image_path = "data/vis/05699-p.png" # 输出处理后图像的路径
    
    # 调用函数进行处理
    invert_mask(input_image_path, output_image_path)
    