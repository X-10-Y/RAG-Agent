import os
from typing import List, Dict
# ==================== PDF 解析 ====================
try:
    import pypdf
except ImportError:
    pypdf = None
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
# ==================== OCR 扫描件支持 ====================
try:
    from cnocr import CnOcr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ cnocr 未安装，扫描件将无法识别。")
    print("   安装命令: pip install cnocr -i https://pypi.tuna.tsinghua.edu.cn/simple")
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️ pdf2image 未安装，无法将 PDF 转为图片。")
    print("   安装命令: pip install pdf2image")
# ==================== PDF 文字提取 ====================
def load_pdf_with_pypdf(filepath: str) -> str:
    """使用 pypdf 读取文字型 PDF"""
    reader = pypdf.PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
def load_pdf_with_pdfplumber(filepath: str) -> str:
    """使用 pdfplumber 读取 PDF（能更好地处理表格）"""
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            # 尝试提取表格
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = " | ".join([str(cell) if cell else "" for cell in row])
                    if row_text.strip():
                        text += row_text + "\n"
    return text
# ==================== OCR 扫描件识别 ====================
def ocr_pdf_with_cnocr(filepath: str, dpi: int = 200) -> str:
    """
    使用 cnocr 识别扫描版 PDF 中的文字
    参数：
        filepath: PDF 文件路径
        dpi: 图片分辨率，默认 200（值越高越清晰，但速度越慢）
    返回：
        识别出的完整文本
    """
    if not OCR_AVAILABLE:
        raise ImportError("请安装 cnocr: pip install cnocr -i https://pypi.tuna.tsinghua.edu.cn/simple")
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("请安装 pdf2image: pip install pdf2image")
    print(f"🔍 正在对 {os.path.basename(filepath)} 进行 OCR 识别（可能较慢）...")
    # 初始化 OCR 引擎
    ocr = CnOcr()
    # 将 PDF 每一页转为图片
    images = convert_from_path(filepath, dpi=dpi)
    full_text = ""
    for i, image in enumerate(images):
        print(f"  识别第 {i+1}/{len(images)} 页...")
        # cnocr 直接接受 PIL Image 对象
        result = ocr.ocr(image)
        if result:
            page_text = '\n'.join([line['text'] for line in result])
            full_text += f"--- 第 {i+1} 页 ---\n{page_text}\n"
    print(f"✅ OCR 完成，共识别 {len(images)} 页")
    return full_text
# ==================== 统一文档加载入口 ====================
def load_document(filepath: str) -> str:
    """根据文件扩展名选择对应的加载器"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        # 第1步：尝试用 pdfplumber 提取文字
        if pdfplumber:
            try:
                text = load_pdf_with_pdfplumber(filepath)
                # 如果提取的文字很少（可能是扫描件），触发 OCR
                if len(text.strip()) < 50:
                    print(f"⚠️ PDF 文字极少（{len(text.strip())}字符），尝试 OCR...")
                    if OCR_AVAILABLE and PDF2IMAGE_AVAILABLE:
                        return ocr_pdf_with_cnocr(filepath)
                    else:
                        print("⚠️ OCR 不可用，返回空文本")
                        return ""
                return text
            except Exception as e:
                print(f"⚠️ pdfplumber 读取失败: {e}")
                # 降级到 pypdf
                if pypdf:
                    text = load_pdf_with_pypdf(filepath)
                    if len(text.strip()) < 50:
                        if OCR_AVAILABLE and PDF2IMAGE_AVAILABLE:
                            return ocr_pdf_with_cnocr(filepath)
                    return text
                else:
                    # 直接尝试 OCR
                    if OCR_AVAILABLE and PDF2IMAGE_AVAILABLE:
                        return ocr_pdf_with_cnocr(filepath)
                    raise
        else:
            # 如果没有 pdfplumber，尝试 pypdf
            if pypdf:
                text = load_pdf_with_pypdf(filepath)
                if len(text.strip()) < 50 and OCR_AVAILABLE and PDF2IMAGE_AVAILABLE:
                    return ocr_pdf_with_cnocr(filepath)
                return text
            else:
                # 直接尝试 OCR
                if OCR_AVAILABLE and PDF2IMAGE_AVAILABLE:
                    return ocr_pdf_with_cnocr(filepath)
                else:
                    raise ImportError("请安装 pypdf 或 pdfplumber 处理 PDF，或安装 cnocr + pdf2image 支持扫描件")
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
# ==================== 批量加载目录下所有文档 ====================
def load_text_files(directory: str) -> List[Dict[str, str]]:
    """加载目录下所有支持的文档（.txt 和 .pdf）"""
    documents = []
    supported_ext = ['.txt', '.pdf']
    if not os.path.exists(directory):
        print(f"⚠️ 目录 {directory} 不存在")
        return documents
    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_ext:
            continue
        filepath = os.path.join(directory, filename)
        try:
            content = load_document(filepath)
            documents.append({
                'filename': filename,
                'content': content
            })
            print(f"✅ 已加载: {filename} ({len(content)} 字符)")
        except Exception as e:
            print(f"❌ 加载失败: {filename} - {e}")
    return documents
# ==================== 独立测试 ====================
if __name__ == '__main__':
    docs = load_text_files('../../data/documents')
    for doc in docs:
        print(f"\n📄 {doc['filename']}: {len(doc['content'])} 字符")
        print(f"   前100字符: {doc['content'][:100]}...")