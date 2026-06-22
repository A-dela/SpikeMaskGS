import cv2
import os

def invert_black_white(image_path, output_path=None):
    """
    颠倒黑白图像的颜色（黑变白，白变黑）
    
    参数:
        image_path: 输入图像的路径（黑白图像）
        output_path: 输出图像的保存路径，若为None则在原文件名后加"_inverted"
    """
    # 读取图像（以灰度模式读取，确保处理的是黑白图像）
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 颠倒黑白（255 - 像素值，黑色(0)变白色(255)，白色(255)变黑色(0)）
    inverted_img = 255 - img
    
    # 处理输出路径
    if output_path is None:
        # 分离文件名和扩展名
        dir_name, file_name = os.path.split(image_path)
        base_name, ext = os.path.splitext(file_name)
        # 构建默认输出路径
        output_path = os.path.join(dir_name, f"{base_name}_inverted{ext}")
    
    # 保存处理后的图像
    cv2.imwrite(output_path, inverted_img)
    print(f"颠倒后的图像已保存至: {output_path}")
    
    return inverted_img

# 示例用法
if __name__ == "__main__":
    # 输入图像路径（替换为你的黑白图像路径）
    input_image = "checkpoint/sear_steak/train/ours_13000/dyn_image/00000.png"
    # 可选：指定输出路径，不指定则使用默认路径
    output_image = "checkpoint/sear_steak/train/ours_13000/dyn_image/00000_inverted.png"  # 或设置为 "output_image.png"
    
    try:
        invert_black_white(input_image, output_image)
    except Exception as e:
        print(f"处理出错: {str(e)}")