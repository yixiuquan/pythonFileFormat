"""
读取pdf文件，并将其中的文字、图片提取放到markdown文件中
安装包：
pip install PyMuPDF markdown
"""
import fitz
import os
import re

def is_page_number(text, bbox, page_height, page_width):
    """判断文本是否为页码"""
    text = text.strip()
    # 如果文本是纯数字
    if text.isdigit():
        # 检查位置是否在页面底部中间
        y_pos = bbox[1]  # y坐标
        x_center = (bbox[0] + bbox[2]) / 2  # x坐标中心点
        
        # 在页面底部20%范围内
        is_bottom = y_pos > page_height * 0.8
        # 在页面水平中间40%范围内
        is_center = page_width * 0.3 < x_center < page_width * 0.7
        
        return is_bottom and is_center
    return False

def get_text_features(span):
    """分析文本的特征"""
    return {
        'size': span['size'],
        'font': span['font'],
        'color': span['color'],
        'flags': span['flags'],  # 包含粗体、斜体等信息
        'text': span['text']
    }

def analyze_text_style(spans):
    """分析一组文本的式特征"""
    features = []
    for span in spans:
        features.append(get_text_features(span))
    
    # 计算主要字体大小
    sizes = [f['size'] for f in features]
    main_size = max(set(sizes), key=sizes.count)
    
    # 改进粗体判断逻辑
    # 1. 检查字体名称中是否包含 Bold 关键字
    # 2. 检查flags中的粗体标志
    # 3. 要求主要文本都是粗体，而不是仅有部分文本是粗体
    total_text_length = sum(len(f['text']) for f in features)
    bold_text_length = sum(
        len(f['text']) 
        for f in features 
        if (f['flags'] & 2) or  # 粗体标志
           ('bold' in f['font'].lower()) or  # 字体名包含bold
           ('heavy' in f['font'].lower())    # 字体名包含heavy
    )
    
    # 只有当超过70%的文本是粗体时才认为整体是粗体
    is_bold = (bold_text_length / total_text_length) > 0.7 if total_text_length > 0 else False
    
    # 判断是否包含斜体
    is_italic = any(f['flags'] & 1 for f in features)  # 1 表示斜体
    
    return {
        'main_size': main_size,
        'is_bold': is_bold,
        'is_italic': is_italic,
        'text': ''.join(f['text'] for f in features)
    }

def is_title(block, font_sizes):
    """判断文本块是否为标题"""
    if not block.get("lines"):
        return False
        
    # 分析第一行的样式特征
    first_line = block["lines"][0]
    text_style = analyze_text_style(first_line["spans"])
    
    # 获取当前块的字体大小
    current_size = text_style['main_size']
    
    # 计算平均字体大小
    avg_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
    
    # 标题特征判断条件
    conditions = [
        # 1. 字体大小明显大于平均值（权重更高）
        (current_size > avg_size * 1.2, 2),  # 赋予权重2
        
        # 2. 使用粗体（权重降低）
        (text_style['is_bold'], 1),
        
        # 3. 文本长度和格式特征（权重更高）
        (len(text_style['text'].strip()) < 100 and 
         not text_style['text'].strip().endswith(('。', '！', '？', '.', '!', '?')), 2),
        
        # 4. 行数较少（权重保持）
        (len(block["lines"]) <= 2, 1),
        
        # 5. 包含标题常见开头词（权重更高）
        (any(text_style['text'].strip().startswith(word) 
             for word in ['第', '章', '节', '标题', '附录', '摘要', '引言', '结论']), 2)
    ]
    
    # 计算加权得分
    total_weight = sum(weight for _, weight in conditions)
    score = sum(weight for condition, weight in conditions if condition)
    
    # 需要达到权重总和的40%才认为是标题
    return score >= total_weight * 0.4

def get_title_level(block, font_sizes):
    """根据文本特征确定标题级别"""
    if not block.get("lines"):
        return 6
        
    text_style = analyze_text_style(block["lines"][0]["spans"])
    current_size = text_style['main_size']
    
    # 按字体大小排序
    sorted_sizes = sorted(set(font_sizes), reverse=True)
    
    try:
        base_level = sorted_sizes.index(current_size) + 1
        
        # 根据其他特征调整级别
        if text_style['is_bold']:
            base_level = max(1, base_level - 1)  # 粗体提升一级
        if text_style['text'].strip().startswith(('第', '章')):
            base_level = 1  # 章节标题设为一级
        elif text_style['text'].strip().startswith(('节', '小节')):
            base_level = 2  # 小节标题设为二级
            
        return min(base_level, 6)
    except ValueError:
        return 6

def find_image_caption(page, image_bbox, blocks):
    """查找图片下方的说明文字"""
    image_bottom = image_bbox[3]
    image_center_x = (image_bbox[0] + image_bbox[2]) / 2
    image_width = image_bbox[2] - image_bbox[0]
    
    # 调整参数
    caption_distance_threshold = 30  # 减小说明文字与图片的最大距离
    horizontal_tolerance = image_width * 0.5  # 水平方向容差设为图片宽度的一半
    
    potential_captions = []
    for block in blocks:
        if block.get("lines"):
            block_bbox = block["bbox"]
            block_top = block_bbox[1]
            block_center_x = (block_bbox[0] + block_bbox[2]) / 2
            
            # 检查文本块是否在图片正下方
            vertical_distance = block_top - image_bottom
            horizontal_distance = abs(block_center_x - image_center_x)
            
            if (0 <= vertical_distance <= caption_distance_threshold and 
                horizontal_distance <= horizontal_tolerance):
                
                text = " ".join(span["text"] for line in block["lines"] 
                              for span in line["spans"]).strip()
                
                # 改进图片说明的识别规则
                is_caption = False
                # 1. 检查特征词
                if any(keyword in text.lower() for keyword in ['图', 'fig', 'figure', '表', 'table']):
                    is_caption = True
                # 2. 检查文本长度和位置
                elif (len(text) < 100 and 
                      vertical_distance < 20 and 
                      block_bbox[2] - block_bbox[0] <= image_width * 1.2):
                    is_caption = True
                # 3. 检查是否以数字开头（可能是图号）
                elif re.match(r'^\d+[.-]', text.strip()):
                    is_caption = True
                
                if is_caption:
                    potential_captions.append((text, block_bbox, vertical_distance))
    
    if potential_captions:
        # 优先选择最近的说明文字
        return min(potential_captions, key=lambda x: x[2])[0:2]
    return None, None

def build_image_groups(images, page):
    """构建图片组，只处理横向相邻的图片"""
    if not images:
        return []
    
    def get_image_info(img_index, img):
        bbox = page.get_image_bbox(img)
        if not bbox:
            return None
        return {
            'index': img_index,
            'bbox': bbox,
            'center_x': (bbox[0] + bbox[2]) / 2,
            'center_y': (bbox[1] + bbox[3]) / 2,
            'width': bbox[2] - bbox[0],
            'height': bbox[3] - bbox[1]
        }
    
    # 获取所有图片的信息
    image_infos = []
    for idx, img in enumerate(images):
        info = get_image_info(idx, img)
        if info:
            image_infos.append(info)
    
    if not image_infos:
        return []
    
    # 按y坐标排序，处理每一行的图片
    image_infos.sort(key=lambda x: x['center_y'])
    
    # 分行处理
    rows = []
    current_row = [image_infos[0]]
    y_tolerance = 20  # 垂直方向的容差值
    
    # 按垂直位置分行，增加容差值
    for i in range(1, len(image_infos)):
        current_img = image_infos[i]
        last_img = current_row[-1]
        
        # 判断是否在同一行，使用更灵活的判断条件
        vertical_diff = abs(current_img['center_y'] - last_img['center_y'])
        if vertical_diff <= y_tolerance:  # 使用固定的容差值
            current_row.append(current_img)
        else:
            # 将当前行按x坐标排序
            current_row.sort(key=lambda x: x['bbox'][0])
            rows.append(current_row)
            current_row = [current_img]
    
    # 添加最后一行
    if current_row:
        current_row.sort(key=lambda x: x['bbox'][0])
        rows.append(current_row)
    
    # 处理每一行中的图片，将水平相邻的图片组合在一起
    groups = []
    horizontal_gap_threshold = 30  # 水平间距阈值
    
    for row in rows:
        current_group = [row[0]]
        
        for i in range(1, len(row)):
            current_img = row[i]
            last_img = current_group[-1]
            
            # 计算水平间距
            horizontal_gap = current_img['bbox'][0] - last_img['bbox'][2]
            
            # 如果间距小于阈值，认为是相邻的
            if horizontal_gap < horizontal_gap_threshold:
                current_group.append(current_img)
            else:
                groups.append(current_group)
                current_group = [current_img]
        
        # 添加最后一组
        if current_group:
            groups.append(current_group)
    
    return groups

def extract_images(page, output_dir):
    """提取页面中的图片及其说明文字"""
    image_list = []
    blocks = page.get_text("dict")["blocks"]
    
    # 获取页面尺寸
    page_width = page.rect.width
    page_height = page.rect.height
    
    # 获取所有图片并构建图片组
    images = list(page.get_images(full=True))
    image_groups = build_image_groups(images, page)
    
    # 处理每个图片组
    for group_index, group in enumerate(image_groups):
        try:
            # 计算组的边界框
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            
            # 计算图片组的边界框，保存原始图片宽度
            for img in group:
                bbox = img['bbox']
                min_x = min(min_x, bbox[0])
                min_y = min(min_y, bbox[1])
                max_x = max(max_x, bbox[2])
                max_y = max(max_y, bbox[3])
            
            # 保存原始图片的宽度范围
            original_min_x = min_x
            original_max_x = max_x
            
            # 添加小边距
            min_x -= 2
            min_y -= 2
            max_x += 2
            max_y += 2
            
            # 确保不包含左侧文字
            left_margin = 5  # 左侧安全边距
            for block in blocks:
                if block.get("lines"):
                    block_bbox = block["bbox"]
                    # 检查文本块是否与图片垂直范围重叠
                    if (block_bbox[3] > min_y and block_bbox[1] < max_y):
                        # 如果文本块在图片左侧，确保留出安全距离
                        if block_bbox[2] < min_x:
                            min_x = max(min_x, block_bbox[2] + left_margin)
            
            # 查找说明文字
            caption_text, caption_bbox = find_image_caption(page, [min_x, min_y, max_x, max_y], blocks)
            
            if caption_bbox:
                # 扩展垂直范围以包含说明文字
                max_y = caption_bbox[3] + 2
                
                # 处理说明文字的水平范围
                caption_center = (caption_bbox[0] + caption_bbox[2]) / 2
                image_center = (original_min_x + original_max_x) / 2
                
                # 如果说明文字的中心点与图片中心点偏移不大，则保持图片原有宽度
                if abs(caption_center - image_center) < (original_max_x - original_min_x) * 0.3:
                    # 保持原有图片宽度，只调整垂直方向
                    min_x = original_min_x - 2
                    max_x = original_max_x + 2
                else:
                    # 说明文字偏移较大，需要适当调整水平范围，但以图片宽度为主
                    image_width = original_max_x - original_min_x
                    min_x = min(original_min_x - 2, caption_bbox[0] - 2)
                    max_x = max(original_max_x + 2, caption_bbox[2] + 2)
                    # 确保不会过度扩展
                    if max_x - min_x > image_width * 1.2:
                        # 如果扩展过大，则回退到以图片为准
                        min_x = original_min_x - 2
                        max_x = original_max_x + 2
            
            # 创建最终的裁剪区域
            clip_bbox = [
                max(0, min_x),
                max(0, min_y),
                min(page_width, max_x),
                min(page_height, max_y)
            ]
            
            # 检查边界框的有效性
            if clip_bbox[2] <= clip_bbox[0] or clip_bbox[3] <= clip_bbox[1]:
                print(f"警告：页面 {page.number + 1} 的图片组 {group_index} 截取区域无效，跳过")
                continue
            
            # 确保边界框尺寸合理
            width = clip_bbox[2] - clip_bbox[0]
            height = clip_bbox[3] - clip_bbox[1]
            if width < 1 or height < 1 or width > page_width or height > page_height:
                print(f"警告：页面 {page.number + 1} 的图片组 {group_index} 尺寸无效，跳过")
                continue
            
            try:
                # 创建高分辨率截图
                # 增加缩放比例以提高图片质量
                zoom = 4.0  # 增加到4倍分辨率
                # 使用 FT_KEEP_ZLIB 参数保持原始图片质量
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, clip=clip_bbox, alpha=False)
                
                # 保存图片时使用更高质量的设置
                image_filename = f"image_{page.number + 1}_{group_index}_group.png"
                image_path = os.path.join(output_dir, "images", image_filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                
                # 使用PNG格式保存，确保最佳质量
                pix.save(image_path, output="png")
                
                # 记录图片信息
                image_list.append({
                    "path": os.path.join("images", image_filename),
                    "bbox": clip_bbox,
                    "caption": caption_text
                })
                
            except Exception as e:
                print(f"警告：保存页面 {page.number + 1} 的图片组 {group_index} 时出错: {str(e)}")
                continue
            
        except Exception as e:
            print(f"警告：处理页面 {page.number + 1} 的图片组 {group_index} 时出错: {str(e)}")
            continue
    
    return image_list

def process_text_block(block, font_sizes, page_height, page_width, image_captions=None):
    """
    处理单个文本块
    
    Args:
        block: 文本块
        font_sizes: 字体大小列表
        page_height: 页面高度
        page_width: 页面宽度
        image_captions: 已被用作图片说明的文本块边界框列表
    """
    if not block.get("lines"):
        return ""
    
    # 检查当前文本块是否已被用作图片说明
    if image_captions:
        block_bbox = block["bbox"]
        for caption_bbox in image_captions:
            # 检查当前文本块是否与任何图片说明重叠
            if (block_bbox[0] >= caption_bbox[0] - 1 and 
                block_bbox[2] <= caption_bbox[2] + 1 and 
                block_bbox[1] >= caption_bbox[1] - 1 and 
                block_bbox[3] <= caption_bbox[3] + 1):
                # 如果重叠，跳过这个文本块
                return ""
    
    text_parts = []
    current_style = None
    
    for line in block["lines"]:
        line_style = analyze_text_style(line["spans"])
        text = line_style['text'].strip()
        
        # 跳过页码
        if is_page_number(text, block["bbox"], page_height, page_width):
            continue
        
        # 处理样式变化
        if current_style and line_style['main_size'] != current_style['main_size']:
            text_parts.append("\n")
        
        # 添加文本
        text_parts.append(text)
        current_style = line_style
    
    return " ".join(text_parts)

def pdf_to_markdown(pdf_path, output_dir):
    """将PDF文件转换为Markdown格式"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开PDF文件
    doc = fitz.open(pdf_path)
    
    # 创建markdown文件
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    markdown_path = os.path.join(output_dir, f"{pdf_name}.md")
    
    with open(markdown_path, 'w', encoding='utf-8') as md_file:
        # 遍历每一页
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 获取页面尺寸
            page_height = page.rect.height
            page_width = page.rect.width
            
            # 先处理图片，获取所有图片说明的位置
            images = extract_images(page, output_dir)
            image_captions = []
            for img in images:
                if img.get("caption") and img.get("bbox"):
                    # 记录图片说明文字的位置
                    image_captions.append(img["bbox"])
            
            # 收集所有文本块的字体大小
            font_sizes = []
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("lines"):
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
            
            # 处理文本，传入图片说明位置列表
            current_text = []
            for block in blocks:
                if block.get("lines"):
                    # 判断是否为标题
                    if is_title(block, font_sizes):
                        level = get_title_level(block, font_sizes)
                        text = process_text_block(block, font_sizes, page_height, page_width, image_captions)
                        if text:  # 只有当文本不为空时才添加
                            current_text.append(f"{'#' * level} {text}\n\n")
                    else:
                        text = process_text_block(block, font_sizes, page_height, page_width, image_captions)
                        if text:  # 只有当文本不为空时才添加
                            current_text.append(f"{text}\n\n")
            
            # 写入文本内容
            md_file.write("".join(current_text))
            
            # 写入图片
            for img in images:
                md_file.write(f"![image]({img['path']})\n\n")
            
            # 页面分隔符
            if page_num < len(doc) - 1:
                md_file.write("---\n\n")
    
    doc.close()
    return markdown_path

def batch_convert_pdfs(input_dir, output_base_dir):
    """
    批量转换文件夹中的PDF文件为Markdown
    
    Args:
        input_dir: 输入文件夹路径，包含PDF文件
        output_base_dir: 输出的基础目录
    """
    # 确保输出基础目录存在
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 获取所有PDF文件
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"在 {input_dir} 中没有找到PDF文件")
        return
    
    # 处理每个PDF文件
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        # 为每个PDF创建独立的输出目录
        pdf_name = os.path.splitext(pdf_file)[0]
        output_dir = os.path.join(output_base_dir, pdf_name)
        
        try:
            markdown_path = pdf_to_markdown(pdf_path, output_dir)
            print(f"成功转换 {pdf_file} -> {markdown_path}")
        except Exception as e:
            print(f"转换 {pdf_file} 时出错: {str(e)}")

if __name__ == '__main__':
    # 示例使用
    input_dir = "D:\\pdfs"  # PDF文件所在文件夹
    output_dir = "D:\\outputs"  # 输出的基础目录
    batch_convert_pdfs(input_dir, output_dir)
