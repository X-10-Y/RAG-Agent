import re
from typing import List, Tuple
# ==================== 表格检测 ====================
def is_table_line(line: str) -> bool:
    """判断一行是否包含表格特征（多个 | 分隔符）"""
    return line.count('|') >= 2
def split_into_segments(lines: List[str]) -> List[Tuple[str, str]]:
    """
    将行列表分割为连续段：'table' 或 'text'
    返回 [(类型, 内容), ...]
    """
    segments = []
    current_type = None
    current_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            # 空行：作为分隔符，结束当前段
            if current_lines:
                segments.append((current_type, '\n'.join(current_lines)))
                current_lines = []
                current_type = None
            continue
        table_flag = is_table_line(line)
        if current_type is None:
            current_type = 'table' if table_flag else 'text'
            current_lines.append(line)
        elif (table_flag and current_type == 'table') or (not table_flag and current_type == 'text'):
            current_lines.append(line)
        else:
            # 类型切换：保存当前段，开始新段
            segments.append((current_type, '\n'.join(current_lines)))
            current_type = 'table' if table_flag else 'text'
            current_lines = [line]
    if current_lines:
        segments.append((current_type, '\n'.join(current_lines)))
    return segments
# ==================== 段落切分 ====================
def split_by_paragraph(text: str, max_chunk_size: int = 500) -> List[str]:
    """按段落切分，段落过长则按句子二次切分"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= max_chunk_size:
            chunks.append(para)
        else:
            sentences = re.split(r'[。！？]', para)
            current = []
            current_len = 0
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if current_len + len(sent) + 1 <= max_chunk_size:
                    current.append(sent)
                    current_len += len(sent) + 1
                else:
                    if current:
                        chunks.append('。'.join(current) + '。')
                    current = [sent]
                    current_len = len(sent) + 1
            if current:
                chunks.append('。'.join(current) + '。')
    return chunks
# ==================== 标题感知切分 ====================
def split_by_heading(text: str, max_chunk_size: int = 500) -> List[str]:
    # 匹配章节标题：一、 二、 三、 1. 2. 第X条 第X章 等（允许前面有空白）
    heading_pattern = r'(^[ \t]*[一二三四五六七八九十百千万]+[、.．]\s*|^[ \t]*第[一二三四五六七八九十百千万]+[章节条款]\s*)'
    lines = text.split('\n')
    chunks = []
    current = []
    current_len = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 检测到新标题
        if re.match(heading_pattern, line):
            if current:
                chunk_text = '\n'.join(current)
                if len(chunk_text) > max_chunk_size:
                    chunks.extend(split_by_paragraph(chunk_text, max_chunk_size))
                else:
                    chunks.append(chunk_text)
                current = []
                current_len = 0
        current.append(line)
        current_len += len(line)
    if current:
        chunk_text = '\n'.join(current)
        if len(chunk_text) > max_chunk_size:
            chunks.extend(split_by_paragraph(chunk_text, max_chunk_size))
        else:
            chunks.append(chunk_text)
    return chunks
# ==================== 智能切分（主入口） ====================
def split_by_structure(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    智能切分：
    1. 按行分割，识别表格块（连续含 | 的行）和普通文本块
    2. 对普通文本块，根据是否有章节标题选择切分方式
    3. 按原始顺序合并所有块
    """
    lines = text.split('\n')
    # 1. 分割为段（table 或 text）
    segments = split_into_segments(lines)
    all_chunks = []
    for seg_type, content in segments:
        if seg_type == 'table':
            # 表格块整体保留，不加切分
            all_chunks.append(content)
        else:
            # 普通文本：检测是否有标题
            heading_pattern = r'(^[一二三四五六七八九十百千万]+[、.．]\s*|^第[一二三四五六七八九十百千万]+[章节条款]\s*)'
            if re.search(heading_pattern, content, re.MULTILINE):
                # 有标题 → 按标题切分
                chunks = split_by_heading(content, max_chunk_size)
            else:
                # 无标题 → 按段落切分
                chunks = split_by_paragraph(content, max_chunk_size)
            all_chunks.extend(chunks)
    return all_chunks
# ==================== 兼容旧接口 ====================
def split_by_paragraph_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """兼容旧接口"""
    return split_by_structure(text, max_chunk_size)
# ==================== 测试入口 ====================
if __name__ == '__main__':
    from rag.loader import load_text_files
    docs = load_text_files('../../data/documents')
    for doc in docs:
        print(f"\n📄 {doc['filename']}")
        chunks = split_by_structure(doc['content'], max_chunk_size=500)
        print(f"  切出 {len(chunks)} 个块")
        for i, chunk in enumerate(chunks[:5]):
            preview = chunk[:80].replace('\n', ' ')
            print(f"  块{i+1}: {preview}...")