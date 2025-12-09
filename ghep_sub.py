#!/usr/bin/env python3
"""
Subtitle Sync Tool - Đồng bộ subtitle tiếng Việt với timing từ subtitle tiếng Anh

Hỗ trợ: .srt và .ass
Matching: Sentence Transformers (embedding) hoặc RapidFuzz (fallback)

Cải tiến v2.0:
- Context matching: Sử dụng câu trước + sau để tăng độ chính xác
- Time-aware matching: Penalty dựa vào độ lệch thời gian
- Order preservation: Ngăn chặn đảo dòng trong cùng một câu
- Split sentence detection: Phát hiện câu bị tách để fix đảo dòng

Author: Claude
Version: 2.0
"""

import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import html

# Import thư viện matching
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using RapidFuzz only.")

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: RapidFuzz not available.")


# ============================================================================
# CẤU HÌNH
# ============================================================================

# Ngưỡng matching
MIN_FUZZ_SIMILARITY = 60  # 0-100
MIN_COSINE_SIMILARITY = 0.5  # 0-1

# Chế độ matching
MATCHING_MODE = "auto"  # "strict", "loose", "auto"

# Context matching - Sử dụng câu trước + sau để tăng độ chính xác
CONTEXT_WEIGHT = 0.7  # Trọng số của context (0-1), cao hơn = context quan trọng hơn

# Time-aware matching - Penalty dựa vào độ lệch thời gian
TIME_PENALTY_FACTOR = 0.05  # Penalty per second of time difference
MAX_TIME_PENALTY = 0.3  # Penalty tối đa

# Order preservation - Ngăn chặn đảo dòng
ORDER_PENALTY = 0.4  # Penalty khi dòng sau đứng trước dòng trước
TIME_WINDOW_SECONDS = 5.0  # Cửa sổ thời gian để xác định "câu liên quan"

# High Confidence Anchors - Chỉ match tin cậy mới làm anchor
HIGH_CONFIDENCE_THRESHOLD = 0.75  # Chỉ match có similarity >= giá trị này mới cập nhật last_matched anchor

# Index Jump Penalty - Phạt khi nhảy cóc quá nhiều dòng (chỉ áp dụng khi similarity không đủ cao)
INDEX_JUMP_PENALTY_FACTOR = 0.01  # Penalty cho mỗi dòng nhảy cóc (0.01 = 1% penalty per line - nhẹ hơn)
MAX_INDEX_JUMP_PENALTY = 0.2  # Penalty tối đa cho index jump
INDEX_JUMP_FREE_THRESHOLD = 5  # Cho phép nhảy 5 dòng mà không bị phạt (do có thể thiếu dòng)
INDEX_JUMP_DISABLE_SIMILARITY = 0.70  # Nếu similarity >= giá trị này, bỏ qua index jump penalty

# Duration Check - Kiểm tra độ dài thời lượng để tránh match sai
MAX_DURATION_RATIO = 3.0  # Hard filter: Lệch >3x → skip candidate (không xét)
DURATION_PENALTY_FACTOR = 0.1  # Soft penalty: 10% per second difference
MAX_DURATION_PENALTY = 0.3  # Penalty tối đa cho duration mismatch

# Time offset detection - Tự động phát hiện và bù offset thời gian
OFFSET_SAMPLE_SIZE = 10  # Số sample để detect offset
OFFSET_FIRST_LINES = 3  # Số dòng đầu để tính offset trung bình
MAX_OFFSET_SECONDS = 20.0  # Offset tối đa cho phép (giây)

# Search window - Giới hạn phạm vi tìm kiếm để tăng tốc và độ chính xác
SEARCH_WINDOW_SECONDS = 5.0  # Chỉ xét VI lines trong khoảng ±5s so với EN

# Pre-splitting - Tách các dòng dài thành nhiều dòng nhỏ để tăng độ chính xác
ENABLE_PRE_SPLITTING = False  # Bật/tắt pre-splitting
SPLIT_ON_NEWLINE = True  # Tách theo xuống dòng (\N, \n)
SPLIT_ON_PUNCTUATION = False  # Tách theo dấu câu (.!?…)
MIN_SPLIT_DURATION = 0.3  # Thời gian tối thiểu cho mỗi phần sau khi tách (giây)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SubtitleEntry:
    """Đại diện cho một dòng subtitle"""
    index: int
    start: float  # Thời gian bắt đầu (giây)
    end: float    # Thời gian kết thúc (giây)
    text: str     # Nội dung
    parent_index: Optional[int] = None  # Index của entry gốc (nếu là virtual entry)
    is_virtual: bool = False  # Đánh dấu đây là virtual entry từ pre-splitting
    
    def __repr__(self):
        virtual_mark = "[V]" if self.is_virtual else ""
        return f"SubtitleEntry{virtual_mark}({self.index}, {self.start:.2f}->{self.end:.2f}, '{self.text[:30]}...')"


# ============================================================================
# PARSE SRT
# ============================================================================

def parse_srt(file_path: str) -> List[SubtitleEntry]:
    """
    Parse file SRT
    
    Format SRT:
    1
    00:00:20,000 --> 00:00:24,400
    Text line 1
    Text line 2
    
    Returns:
        List[SubtitleEntry]: Danh sách các dòng subtitle
    """
    entries = []
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    # Tách các khối subtitle (phân cách bởi dòng trống)
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # Dòng 1: index
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        
        # Dòng 2: timing
        timing_match = re.match(
            r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
            lines[1]
        )
        
        if not timing_match:
            continue
        
        # Parse start time
        start = (
            int(timing_match.group(1)) * 3600 +  # hours
            int(timing_match.group(2)) * 60 +    # minutes
            int(timing_match.group(3)) +         # seconds
            int(timing_match.group(4)) / 1000    # milliseconds
        )
        
        # Parse end time
        end = (
            int(timing_match.group(5)) * 3600 +
            int(timing_match.group(6)) * 60 +
            int(timing_match.group(7)) +
            int(timing_match.group(8)) / 1000
        )
        
        # Dòng 3+: text (có thể multiline)
        text = '\n'.join(lines[2:])
        
        entries.append(SubtitleEntry(index, start, end, text))
    
    return entries


# ============================================================================
# PARSE ASS
# ============================================================================

def parse_ass(file_path: str) -> List[SubtitleEntry]:
    """
    Parse file ASS/SSA
    
    Format ASS:
    [Events]
    Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
    Dialogue: 0,0:00:20.00,0:00:24.40,Default,,0,0,0,,Text here
    
    Returns:
        List[SubtitleEntry]: Danh sách các dòng subtitle
    """
    entries = []
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    in_events = False
    format_indices = {}
    
    for line in lines:
        line = line.strip()
        
        # Tìm section [Events]
        if line.startswith('[Events]'):
            in_events = True
            continue
        
        # Kết thúc section
        if line.startswith('[') and in_events:
            break
        
        if not in_events:
            continue
        
        # Parse Format line để biết thứ tự các trường
        if line.startswith('Format:'):
            format_line = line[7:].strip()
            fields = [f.strip() for f in format_line.split(',')]
            for i, field in enumerate(fields):
                format_indices[field] = i
            continue
        
        # Parse Dialogue line
        if line.startswith('Dialogue:'):
            parts = line[9:].strip().split(',', len(format_indices) - 1)
            
            if len(parts) < len(format_indices):
                continue
            
            try:
                # Lấy Start, End, Text
                start_str = parts[format_indices.get('Start', 1)]
                end_str = parts[format_indices.get('End', 2)]
                text = parts[format_indices.get('Text', -1)]
                
                # Parse time (format: H:MM:SS.CS)
                start = parse_ass_time(start_str)
                end = parse_ass_time(end_str)
                
                # Xoá ASS override tags {...}
                text = remove_ass_tags(text)
                
                # Thay \N và \n bằng newline thực
                text = text.replace('\\N', '\n').replace('\\n', '\n')
                
                entries.append(SubtitleEntry(len(entries) + 1, start, end, text))
                
            except (ValueError, IndexError):
                continue
    
    return entries


def parse_ass_time(time_str: str) -> float:
    """
    Parse ASS time format H:MM:SS.CS thành giây
    
    Args:
        time_str: Chuỗi thời gian (vd: "0:00:20.00")
    
    Returns:
        float: Thời gian tính bằng giây
    """
    match = re.match(r'(\d+):(\d{2}):(\d{2})\.(\d{2})', time_str)
    if not match:
        raise ValueError(f"Invalid ASS time format: {time_str}")
    
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    centiseconds = int(match.group(4))
    
    return hours * 3600 + minutes * 60 + seconds + centiseconds / 100


def remove_ass_tags(text: str) -> str:
    """
    Xoá các ASS override tags {...}
    
    Args:
        text: Text có chứa tags
    
    Returns:
        str: Text đã xoá tags
    """
    return re.sub(r'\{[^}]*\}', '', text)


# ============================================================================
# NORMALIZE TEXT
# ============================================================================

# ============================================================================
# PRE-SPLITTING
# ============================================================================

def split_text_by_delimiters(text: str) -> List[str]:
    r"""
    Tách text thành các phần dựa trên:
    - Xuống dòng (\N, \n)
    - Dấu câu (.!?…) theo sau bởi khoảng trắng hoặc end of string
    
    Args:
        text: Text cần tách
    
    Returns:
        List[str]: Danh sách các phần text
    """
    parts = []
    
    # Bước 1: Tách theo xuống dòng nếu enabled
    if SPLIT_ON_NEWLINE:
        # Tách theo \N (ASS) và \n (SRT)
        temp_parts = re.split(r'\\N|\n', text)
    else:
        temp_parts = [text]
    
    # Bước 2: Với mỗi phần, tiếp tục tách theo dấu câu nếu enabled
    if SPLIT_ON_PUNCTUATION:
        for part in temp_parts:
            # Tách theo dấu câu: .!?… theo sau bởi space hoặc end
            # Giữ lại dấu câu trong phần text
            sub_parts = re.split(r'([.!?…]+(?:\s+|$))', part)
            
            # Gộp lại: [text, delimiter, text, delimiter, ...] -> [text+delimiter, text+delimiter, ...]
            i = 0
            while i < len(sub_parts):
                if i + 1 < len(sub_parts) and sub_parts[i+1].strip():
                    # Có delimiter theo sau
                    combined = sub_parts[i] + sub_parts[i+1]
                    if combined.strip():
                        parts.append(combined.strip())
                    i += 2
                else:
                    # Không có delimiter hoặc là phần cuối
                    if sub_parts[i].strip():
                        parts.append(sub_parts[i].strip())
                    i += 1
    else:
        parts = [p.strip() for p in temp_parts if p.strip()]
    
    # Filter empty parts
    parts = [p for p in parts if p.strip()]
    
    return parts if parts else [text]  # Fallback: trả về text gốc nếu không tách được


def split_subtitle_entry(
    entry: SubtitleEntry,
    base_index: int
) -> List[SubtitleEntry]:
    """
    Tách một subtitle entry thành nhiều virtual entries
    
    Args:
        entry: Entry gốc
        base_index: Index bắt đầu cho các virtual entries
    
    Returns:
        List[SubtitleEntry]: Danh sách virtual entries (có thể chỉ có 1 nếu không tách được)
    """
    # Tách text
    text_parts = split_text_by_delimiters(entry.text)
    
    # Nếu chỉ có 1 phần, không cần split
    if len(text_parts) <= 1:
        return [entry]
    
    # Tính tổng độ dài text (không kể khoảng trắng)
    total_length = sum(len(p.replace(' ', '')) for p in text_parts)
    if total_length == 0:
        return [entry]
    
    # Tính thời lượng của entry gốc
    total_duration = entry.end - entry.start
    
    # Tạo virtual entries
    virtual_entries = []
    current_time = entry.start
    
    for i, text_part in enumerate(text_parts):
        # Tính tỉ lệ độ dài của phần này
        part_length = len(text_part.replace(' ', ''))
        ratio = part_length / total_length
        
        # Tính thời gian cho phần này
        part_duration = total_duration * ratio
        
        # Đảm bảo độ dài tối thiểu
        if part_duration < MIN_SPLIT_DURATION:
            part_duration = MIN_SPLIT_DURATION
        
        # Tính start/end time
        part_start = current_time
        part_end = min(current_time + part_duration, entry.end)
        
        # Nếu là phần cuối, đảm bảo end time = entry.end
        if i == len(text_parts) - 1:
            part_end = entry.end
        
        # Tạo virtual entry
        virtual_entry = SubtitleEntry(
            index=base_index + i,
            start=part_start,
            end=part_end,
            text=text_part,
            parent_index=entry.index,
            is_virtual=True
        )
        virtual_entries.append(virtual_entry)
        
        current_time = part_end
    
    return virtual_entries


def apply_pre_splitting(
    entries: List[SubtitleEntry],
    entry_type: str = "unknown"
) -> List[SubtitleEntry]:
    """
    Áp dụng pre-splitting cho toàn bộ danh sách entries
    
    Args:
        entries: Danh sách entries gốc
        entry_type: Loại entry ("EN" hoặc "VI") cho logging
    
    Returns:
        List[SubtitleEntry]: Danh sách entries sau khi split (có thể nhiều hơn)
    """
    if not ENABLE_PRE_SPLITTING:
        return entries
    
    print(f"  Pre-splitting {entry_type} entries...")
    
    virtual_entries = []
    base_index = 1  # Index cho virtual entries
    split_count = 0
    
    for entry in entries:
        splits = split_subtitle_entry(entry, base_index)
        virtual_entries.extend(splits)
        base_index += len(splits)
        
        if len(splits) > 1:
            split_count += 1
    
    print(f"    Original: {len(entries)} entries")
    print(f"    After split: {len(virtual_entries)} entries ({split_count} entries were split)")
    
    return virtual_entries


# ============================================================================
# NORMALIZE TEXT
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Chuẩn hoá text để so sánh
    - Lowercase
    - Xoá punctuation
    - Xoá HTML tags
    - Xoá ASS override tags
    - Xoá newline
    - Xoá khoảng trắng thừa
    
    Args:
        text: Text cần chuẩn hoá
    
    Returns:
        str: Text đã chuẩn hoá
    """
    # Xoá HTML entities
    text = html.unescape(text)
    
    # Xoá HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Xoá ASS tags
    text = remove_ass_tags(text)
    
    # Xoá newline
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Lowercase
    text = text.lower()
    
    # Xoá punctuation (giữ lại chữ cái, số, khoảng trắng)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Xoá khoảng trắng thừa
    text = ' '.join(text.split())
    
    return text


# ============================================================================
# TIME OFFSET DETECTION
# ============================================================================

class TimeOffsetDetector:
    """
    Tự động phát hiện và bù offset thời gian giữa subtitle EN và VI.
    
    Giải quyết trường hợp:
    - Sub VI bị trễ/sớm so với sub EN (offset toàn cục)
    - Offset thay đổi theo thời lượng video (piecewise offset)
    """
    
    def __init__(self):
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('intfloat/multilingual-e5-small')
            except:
                pass
    
    def _quick_similarity(self, text1: str, text2: str) -> float:
        """
        Tính similarity nhanh giữa 2 text (dùng cho offset detection)
        Ưu tiên RapidFuzz vì nhanh hơn embedding
        """
        norm1 = normalize_text(text1)
        norm2 = normalize_text(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        if RAPIDFUZZ_AVAILABLE:
            return fuzz.ratio(norm1, norm2) / 100
        elif self.model is not None:
            emb1 = self.model.encode([norm1])[0]
            emb2 = self.model.encode([norm2])[0]
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        
        return 0.0
    
    def _find_best_offset_for_sample(
        self,
        en_entry: SubtitleEntry,
        vi_entries: List[SubtitleEntry],
        search_range: float = MAX_OFFSET_SECONDS
    ) -> Tuple[float, float]:
        """
        Tìm offset tốt nhất cho một entry EN bằng cách thử các VI gần đó
        
        Returns:
            Tuple[offset, similarity]: (vi_start - en_start, best_similarity)
        """
        en_time = en_entry.start
        best_offset = 0.0
        best_sim = 0.0
        
        for vi_entry in vi_entries:
            # Chỉ xét các VI trong search range
            time_diff = vi_entry.start - en_time
            if abs(time_diff) > search_range:
                continue
            
            sim = self._quick_similarity(en_entry.text, vi_entry.text)
            if sim > best_sim:
                best_sim = sim
                best_offset = time_diff
        
        return best_offset, best_sim
    
    def detect_global_offset(
        self,
        en_entries: List[SubtitleEntry],
        vi_entries: List[SubtitleEntry]
    ) -> float:
        """
        Phát hiện offset toàn cục bằng cách sample và vote
        
        Args:
            en_entries: Danh sách subtitle tiếng Anh
            vi_entries: Danh sách subtitle tiếng Việt
        
        Returns:
            float: Offset (giây). Dương = VI trễ hơn EN
        """
        if not en_entries or not vi_entries:
            return 0.0
        
        # Sample đều từ đầu đến cuối
        sample_indices = []
        step = max(1, len(en_entries) // OFFSET_SAMPLE_SIZE)
        for i in range(0, len(en_entries), step):
            sample_indices.append(i)
            if len(sample_indices) >= OFFSET_SAMPLE_SIZE:
                break
        
        # Tìm offset cho từng sample
        offsets = []
        weights = []  # Weight = similarity (cao hơn = tin cậy hơn)
        
        for idx in sample_indices:
            offset, sim = self._find_best_offset_for_sample(
                en_entries[idx], vi_entries
            )
            if sim > 0.4:  # Chỉ lấy những match đủ tốt
                offsets.append(offset)
                weights.append(sim)
        
        if not offsets:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        weighted_offset = sum(o * w for o, w in zip(offsets, weights)) / total_weight
        
        return weighted_offset
    
    def detect_piecewise_offsets(
        self,
        en_entries: List[SubtitleEntry],
        vi_entries: List[SubtitleEntry]
    ) -> float:
        """
        Tính offset trung bình từ vài dòng đầu tiên
        (Đơn giản hóa: không chia thành nhiều đoạn nữa)
        
        Returns:
            float: Offset trung bình (giây)
        """
        if not en_entries or not vi_entries:
            return 0.0
        
        # Lấy N dòng đầu tiên
        num_lines = min(OFFSET_FIRST_LINES, len(en_entries), len(vi_entries))
        
        offsets = []
        weights = []
        
        for i in range(num_lines):
            en_entry = en_entries[i]
            offset, sim = self._find_best_offset_for_sample(en_entry, vi_entries)
            
            if sim > 0.4:  # Chỉ lấy những match đủ tốt
                offsets.append(offset)
                weights.append(sim)
        
        if not offsets:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        avg_offset = sum(o * w for o, w in zip(offsets, weights)) / total_weight
        return avg_offset
    
    def apply_offset(
        self,
        vi_entries: List[SubtitleEntry],
        offset: float
    ) -> List[SubtitleEntry]:
        """
        Áp dụng offset cho tất cả VI entries
        
        Args:
            vi_entries: Danh sách subtitle tiếng Việt
            offset: Offset (giây). Dương = VI trễ, cần trừ đi
        
        Returns:
            List[SubtitleEntry]: Entries đã điều chỉnh thời gian
        """
        adjusted = []
        for entry in vi_entries:
            adjusted.append(SubtitleEntry(
                index=entry.index,
                start=entry.start - offset,  # Trừ offset để căn chỉnh
                end=entry.end - offset,
                text=entry.text
            ))
        return adjusted
    
    def auto_detect_and_apply(
        self,
        en_entries: List[SubtitleEntry],
        vi_entries: List[SubtitleEntry],
        use_piecewise: bool = True
    ) -> Tuple[List[SubtitleEntry], dict]:
        """
        Tự động phát hiện và áp dụng offset
        
        Args:
            en_entries: Subtitle tiếng Anh
            vi_entries: Subtitle tiếng Việt
            use_piecewise: Sử dụng piecewise offset (True) hay global offset (False)
        
        Returns:
            Tuple[adjusted_vi_entries, info_dict]
        """
        info = {
            'global_offset': 0.0,
            'piecewise_offset': 0.0,
            'method': 'none'
        }
        
        # Detect global offset trước
        global_offset = self.detect_global_offset(en_entries, vi_entries)
        info['global_offset'] = global_offset
        
        # Nếu offset nhỏ (< 1 giây), không cần bù
        if abs(global_offset) < 1.0:
            info['method'] = 'none (offset < 1s)'
            return vi_entries, info
        
        if use_piecewise:
            # Detect piecewise (từ first lines)
            piecewise_offset = self.detect_piecewise_offsets(en_entries, vi_entries)
            info['piecewise_offset'] = piecewise_offset
            
            # So sánh: nếu piecewise và global khác nhau đáng kể, ưu tiên piecewise
            if abs(piecewise_offset - global_offset) > 1.0:
                info['method'] = 'piecewise (first lines)'
                return self.apply_offset(vi_entries, piecewise_offset), info
        
        # Áp dụng global offset
        info['method'] = 'global'
        return self.apply_offset(vi_entries, global_offset), info


# ============================================================================
# MATCHING
# ============================================================================

class SubtitleMatcher:
    """Class để matching subtitle tiếng Việt với tiếng Anh
    
    Cải tiến:
    - Context matching: Sử dụng câu trước + sau để tăng độ chính xác
    - Time-aware matching: Penalty dựa vào độ lệch thời gian
    - Order preservation: Ngăn chặn đảo dòng trong cùng một câu
    """
    
    def __init__(self, mode: str = "auto"):
        """
        Khởi tạo matcher
        
        Args:
            mode: Chế độ matching ("strict", "loose", "auto")
        """
        self.mode = mode
        self.model = None
        
        # Thử load Sentence Transformer model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print("Loading Sentence Transformer model...")
                # Sử dụng multilingual-e5-small cho embedding tốt hơn
                self.model = SentenceTransformer('intfloat/multilingual-e5-small')
                print("✓ Sentence Transformer loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load Sentence Transformer: {e}")
                self.model = None
        
        # Điều chỉnh ngưỡng theo mode
        if mode == "strict":
            self.min_fuzz = MIN_FUZZ_SIMILARITY + 10
            self.min_cosine = MIN_COSINE_SIMILARITY + 0.1
        elif mode == "loose":
            self.min_fuzz = max(MIN_FUZZ_SIMILARITY - 15, 40)
            self.min_cosine = max(MIN_COSINE_SIMILARITY - 0.15, 0.3)
        else:  # auto
            self.min_fuzz = MIN_FUZZ_SIMILARITY
            self.min_cosine = MIN_COSINE_SIMILARITY
        
        # Lưu trữ context đã chuẩn hoá
        self._en_contexts = {}
        self._vi_contexts = {}
        # Lưu trữ embeddings đã tính trước
        self._en_embeddings = {}
        self._vi_embeddings = {}
        self._en_context_embeddings = {}
        self._vi_context_embeddings = {}
        # Lưu trữ normalized texts cho RapidFuzz fallback
        self._en_normalized = {}
        self._vi_normalized = {}
        # Lưu trữ last matched vi index để enforce order
        self._last_matched_vi_idx = -1
        self._last_matched_en_start_time = -1.0
    
    def _build_context(
        self,
        entries: List[SubtitleEntry],
        idx: int
    ) -> str:
        """
        Xây dựng context string từ câu trước + hiện tại + sau
        
        Args:
            entries: Danh sách subtitle
            idx: Index của câu hiện tại
        
        Returns:
            str: Context string đã normalize
        """
        parts = []
        
        # Câu trước
        if idx > 0:
            parts.append(normalize_text(entries[idx - 1].text))
        
        # Câu hiện tại
        parts.append(normalize_text(entries[idx].text))
        
        # Câu sau
        if idx < len(entries) - 1:
            parts.append(normalize_text(entries[idx + 1].text))
        
        return " | ".join(filter(None, parts))
    
    def _precompute_contexts(
        self,
        en_entries: List[SubtitleEntry],
        vi_entries: List[SubtitleEntry]
    ):
        """
        Tính trước context cho tất cả entries để tối ưu performance
        """
        print("  Pre-computing contexts...")
        
        for i, entry in enumerate(en_entries):
            self._en_contexts[entry.index] = self._build_context(en_entries, i)
        
        for i, entry in enumerate(vi_entries):
            self._vi_contexts[entry.index] = self._build_context(vi_entries, i)
    
    def _precompute_embeddings(
        self,
        en_entries: List[SubtitleEntry],
        vi_entries: List[SubtitleEntry]
    ):
        """
        Pre-compute tất cả embeddings để tối ưu performance.
        Batch encoding thay vì encode từng text riêng lẻ.
        """
        if self.model is None:
            return
        
        print("  Pre-computing embeddings (batch)...")
        
        # Collect all texts
        en_texts = []
        en_text_indices = []  # Map to entry index
        for entry in en_entries:
            text = normalize_text(entry.text)
            if text:
                en_texts.append(text)
                en_text_indices.append(entry.index)
        
        vi_texts = []
        vi_text_indices = []
        for entry in vi_entries:
            text = normalize_text(entry.text)
            if text:
                vi_texts.append(text)
                vi_text_indices.append(entry.index)
        
        # Batch encode texts
        if en_texts:
            en_embeddings = self.model.encode(en_texts, show_progress_bar=False)
            for i, idx in enumerate(en_text_indices):
                self._en_embeddings[idx] = en_embeddings[i]
        
        if vi_texts:
            vi_embeddings = self.model.encode(vi_texts, show_progress_bar=False)
            for i, idx in enumerate(vi_text_indices):
                self._vi_embeddings[idx] = vi_embeddings[i]
        
        # Batch encode contexts if needed
        if CONTEXT_WEIGHT > 0:
            print("  Pre-computing context embeddings...")
            en_contexts = [self._en_contexts.get(idx, "") for idx in en_text_indices]
            vi_contexts = [self._vi_contexts.get(idx, "") for idx in vi_text_indices]
            
            # Filter empty contexts
            en_ctx_valid = [(i, ctx) for i, ctx in zip(en_text_indices, en_contexts) if ctx]
            vi_ctx_valid = [(i, ctx) for i, ctx in zip(vi_text_indices, vi_contexts) if ctx]
            
            if en_ctx_valid:
                en_ctx_embeddings = self.model.encode(
                    [ctx for _, ctx in en_ctx_valid], show_progress_bar=False
                )
                for j, (idx, _) in enumerate(en_ctx_valid):
                    self._en_context_embeddings[idx] = en_ctx_embeddings[j]
            
            if vi_ctx_valid:
                vi_ctx_embeddings = self.model.encode(
                    [ctx for _, ctx in vi_ctx_valid], show_progress_bar=False
                )
                for j, (idx, _) in enumerate(vi_ctx_valid):
                    self._vi_context_embeddings[idx] = vi_ctx_embeddings[j]
        
        print(f"  ✓ Pre-computed {len(self._en_embeddings)} EN + {len(self._vi_embeddings)} VI embeddings")
    
    def _calculate_time_penalty(
        self,
        en_entry: SubtitleEntry,
        vi_entry: SubtitleEntry
    ) -> float:
        """
        Tính time penalty dựa vào độ lệch thời gian
        
        Args:
            en_entry: Dòng tiếng Anh
            vi_entry: Dòng tiếng Việt
        
        Returns:
            float: Penalty (0 đến MAX_TIME_PENALTY)
        """
        time_diff = abs(vi_entry.start - en_entry.start)
        penalty = time_diff * TIME_PENALTY_FACTOR
        return min(penalty, MAX_TIME_PENALTY)
    
    def _calculate_order_penalty(
        self,
        vi_entry: SubtitleEntry,
        en_entry: SubtitleEntry
    ) -> float:
        """
        Tính order penalty để ngăn chặn đảo dòng
        
        Nếu câu tiếng Việt đứng TRƯỚC câu đã match trước đó
        và câu tiếng Anh hiện tại gần với câu trước đó về thời gian,
        thì đây có thể là trường hợp đảo dòng → áp dụng penalty
        
        Args:
            vi_entry: Dòng tiếng Việt candidate
            en_entry: Dòng tiếng Anh hiện tại
        
        Returns:
            float: Penalty (0 hoặc ORDER_PENALTY)
        """
        if self._last_matched_vi_idx < 0:
            return 0.0
        
        # Nếu vi_entry đứng TRƯỚC câu đã match
        if vi_entry.index < self._last_matched_vi_idx:
            # Kiểm tra xem có trong cùng time window không
            time_from_last = en_entry.start - self._last_matched_en_start_time
            if 0 <= time_from_last <= TIME_WINDOW_SECONDS:
                # Đây có thể là trường hợp đảo dòng → penalty
                return ORDER_PENALTY
        
        return 0.0
    
    def _calculate_index_jump_penalty(
        self,
        vi_entry: SubtitleEntry,
        similarity: float
    ) -> float:
        """
        Tính penalty dựa trên khoảng cách index (nhảy cóc)
        
        CHÚ Ý: Chỉ áp dụng penalty nhẹ khi similarity không đủ cao.
        Ưu tiên SIMILARITY làm yếu tố quyết định chính.
        
        Args:
            vi_entry: Dòng tiếng Việt candidate
            similarity: Similarity score của match này
        
        Returns:
            float: Penalty dựa trên khoảng cách index (0 đến MAX_INDEX_JUMP_PENALTY)
        """
        if self._last_matched_vi_idx < 0:
            return 0.0
        
        # Nếu similarity cao, bỏ qua index jump penalty
        # (Vì match tốt vẫn quan trọng hơn dù nhảy xa)
        if similarity >= INDEX_JUMP_DISABLE_SIMILARITY:
            return 0.0
        
        # Vị trí dự kiến: ngay sau câu vừa match
        expected_idx = self._last_matched_vi_idx + 1
        
        # Khoảng cách so với vị trí dự kiến
        distance = abs(vi_entry.index - expected_idx)
        
        # Cho phép nhảy INDEX_JUMP_FREE_THRESHOLD dòng mà không bị phạt
        # (do subtitle Việt có thể thiếu dòng so với Anh)
        if distance <= INDEX_JUMP_FREE_THRESHOLD:
            return 0.0
        
        # Tính penalty: càng xa càng phạt nặng (nhưng nhẹ hơn trước)
        penalty = (distance - INDEX_JUMP_FREE_THRESHOLD) * INDEX_JUMP_PENALTY_FACTOR
        return min(penalty, MAX_INDEX_JUMP_PENALTY)
    
    def _calculate_duration_penalty(
        self,
        en_entry: SubtitleEntry,
        vi_entry: SubtitleEntry
    ) -> float:
        """
        Tính penalty dựa trên độ lệch thời lượng (duration)
        
        Dùng để phạt các cặp có duration chênh lệch quá lớn.
        Ví dụ: EN dài 1s nhưng VI dài 5s → có thể match sai
        
        Args:
            en_entry: Dòng tiếng Anh
            vi_entry: Dòng tiếng Việt
        
        Returns:
            float: Penalty dựa trên độ lệch duration (0 đến MAX_DURATION_PENALTY)
        """
        en_duration = en_entry.end - en_entry.start
        vi_duration = vi_entry.end - vi_entry.start
        
        # Tính độ chênh lệch duration (giây)
        duration_diff = abs(en_duration - vi_duration)
        
        # Tính penalty: càng chênh lệch càng phạt nặng
        penalty = duration_diff * DURATION_PENALTY_FACTOR
        
        return min(penalty, MAX_DURATION_PENALTY)
    
    def _get_similarity_score(
        self,
        en_idx: int,
        vi_idx: int
    ) -> float:
        """
        Lấy similarity score từ pre-computed embeddings (hoặc dùng RapidFuzz fallback)
        
        Args:
            en_idx: Index của entry tiếng Anh
            vi_idx: Index của entry tiếng Việt
        
        Returns:
            float: Similarity score (0-1)
        """
        # Nếu có embeddings
        if self.model is not None:
            en_emb = self._en_embeddings.get(en_idx)
            vi_emb = self._vi_embeddings.get(vi_idx)
            
            if en_emb is None or vi_emb is None:
                return 0.0
            
            # Text similarity
            text_score = float(np.dot(en_emb, vi_emb) / (
                np.linalg.norm(en_emb) * np.linalg.norm(vi_emb) + 1e-9
            ))
            
            # Context similarity
            if CONTEXT_WEIGHT > 0:
                en_ctx_emb = self._en_context_embeddings.get(en_idx)
                vi_ctx_emb = self._vi_context_embeddings.get(vi_idx)
                
                if en_ctx_emb is not None and vi_ctx_emb is not None:
                    context_score = float(np.dot(en_ctx_emb, vi_ctx_emb) / (
                        np.linalg.norm(en_ctx_emb) * np.linalg.norm(vi_ctx_emb) + 1e-9
                    ))
                    return (1 - CONTEXT_WEIGHT) * text_score + CONTEXT_WEIGHT * context_score
            
            return text_score
        
        # Fallback: RapidFuzz
        elif RAPIDFUZZ_AVAILABLE:
            en_norm = self._en_normalized.get(en_idx, "")
            vi_norm = self._vi_normalized.get(vi_idx, "")
            
            if not en_norm or not vi_norm:
                return 0.0
            
            text_score = fuzz.ratio(en_norm, vi_norm) / 100
            
            if CONTEXT_WEIGHT > 0:
                en_ctx = self._en_contexts.get(en_idx, "")
                vi_ctx = self._vi_contexts.get(vi_idx, "")
                if en_ctx and vi_ctx:
                    context_score = fuzz.ratio(en_ctx, vi_ctx) / 100
                    return (1 - CONTEXT_WEIGHT) * text_score + CONTEXT_WEIGHT * context_score
            
            return text_score
        
        return 0.0
    
    def find_best_match(
        self,
        en_entry: SubtitleEntry,
        en_idx: int,
        en_entries: List[SubtitleEntry],
        vi_entries: List[SubtitleEntry],
        used_indices: set
    ) -> Optional[Tuple[SubtitleEntry, float]]:
        """
        Tìm dòng tiếng Việt khớp nhất với dòng tiếng Anh
        
        Args:
            en_entry: Dòng tiếng Anh
            en_idx: Index của dòng tiếng Anh trong list
            en_entries: Danh sách tất cả dòng tiếng Anh
            vi_entries: Danh sách dòng tiếng Việt
            used_indices: Set các index tiếng Việt đã được dùng
        
        Returns:
            Tuple[SubtitleEntry, float] hoặc None: (entry khớp, similarity score)
        """
        # Check if en_entry has embedding
        if en_entry.index not in self._en_embeddings and en_entry.index not in self._en_normalized:
            return None
        
        best_match = None
        best_score = -1.0
        
        candidates = []
        
        # Phase 1: Thu thập candidates với scores
        # Chỉ xét VI entries trong search window để tăng tốc và độ chính xác
        en_start_time = en_entry.start
        en_duration = en_entry.end - en_entry.start
        
        for vi_idx, vi_entry in enumerate(vi_entries):
            if vi_entry.index in used_indices:
                continue
            
            # Search window: chỉ xét VI entries gần EN về thời gian
            time_diff = abs(vi_entry.start - en_start_time)
            if time_diff > SEARCH_WINDOW_SECONDS:
                continue
            
            # Duration Hard Filter: Loại bỏ candidates có duration lệch quá xa
            vi_duration = vi_entry.end - vi_entry.start
            if en_duration > 0 and vi_duration > 0:  # Tránh chia cho 0
                duration_ratio = max(en_duration, vi_duration) / min(en_duration, vi_duration)
                if duration_ratio > MAX_DURATION_RATIO:
                    # Lệch quá xa (>3x) → skip candidate này
                    continue
            
            # Check if vi_entry has embedding
            if vi_entry.index not in self._vi_embeddings and vi_entry.index not in self._vi_normalized:
                continue
            
            # Tính similarity score từ pre-computed embeddings
            similarity = self._get_similarity_score(en_entry.index, vi_entry.index)
            
            # Trừ time penalty
            time_penalty = self._calculate_time_penalty(en_entry, vi_entry)
            
            # Trừ order penalty
            order_penalty = self._calculate_order_penalty(vi_entry, en_entry)
            
            # Trừ index jump penalty (chỉ khi similarity không đủ cao)
            index_penalty = self._calculate_index_jump_penalty(vi_entry, similarity)
            
            # Trừ duration penalty (soft penalty cho cặp lệch vừa phải)
            duration_penalty = self._calculate_duration_penalty(en_entry, vi_entry)
            
            final_score = similarity - time_penalty - order_penalty - index_penalty - duration_penalty
            
            candidates.append((vi_entry, vi_idx, final_score, similarity))
        
        if not candidates:
            return None
        
        # Sắp xếp theo final_score giảm dần
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Lấy candidate tốt nhất
        best_vi, best_vi_idx, best_final, best_raw = candidates[0]
        
        # Kiểm tra ngưỡng (dùng raw similarity để so sánh với threshold)
        if self.model is not None:
            threshold = self.min_cosine
        else:
            threshold = self.min_fuzz / 100
        
        if best_raw >= threshold:
            best_match = best_vi
            best_score = best_final
        
        if best_match:
            return (best_match, best_score)
        
        return None
    
    def _detect_split_sentences(
        self,
        en_entries: List[SubtitleEntry]
    ) -> List[Tuple[int, int]]:
        """
        Phát hiện các cặp câu tiếng Anh có thể là một câu bị tách
        
        Heuristics:
        - Thời gian kết thúc câu 1 rất gần với thời gian bắt đầu câu 2
        - Câu 1 không kết thúc bằng dấu chấm hoặc dấu hỏi
        - Câu 2 bắt đầu bằng chữ thường
        
        Returns:
            List[Tuple[int, int]]: Danh sách (index1, index2) các câu bị split
        """
        split_pairs = []
        
        for i in range(len(en_entries) - 1):
            curr = en_entries[i]
            next_entry = en_entries[i + 1]
            
            # Time gap rất nhỏ (< 0.5 giây)
            time_gap = next_entry.start - curr.end
            if time_gap > 0.5:
                continue
            
            # Câu hiện tại không kết thúc câu hoàn chỉnh
            curr_text = curr.text.strip()
            if curr_text.endswith(('.', '?', '!', '"', '"', "'")):
                continue
            
            # Câu tiếp theo bắt đầu bằng chữ thường (không phải đầu câu mới)
            next_text = next_entry.text.strip()
            if not next_text:
                continue
            
            # Nếu bắt đầu bằng chữ thường → có thể là split
            first_char = next_text[0]
            if first_char.islower() or first_char in ',;:':
                split_pairs.append((i, i + 1))
        
        return split_pairs
    
    def match_subtitles(
        self,
        en_entries: List[SubtitleEntry],
        vi_entries: List[SubtitleEntry]
    ) -> List[Tuple[SubtitleEntry, Optional[SubtitleEntry], float]]:
        """
        Match tất cả các dòng subtitle
        
        Thuật toán cải tiến:
        1. Pre-compute contexts cho tất cả entries
        2. Detect split sentences trong tiếng Anh
        3. Match từng dòng với time-aware và order preservation
        4. Post-process để fix các trường hợp đảo dòng
        
        Args:
            en_entries: Danh sách dòng tiếng Anh
            vi_entries: Danh sách dòng tiếng Việt
        
        Returns:
            List[Tuple]: [(en_entry, vi_entry_or_none, score), ...]
        """
        results = []
        used_indices = set()
        
        print(f"\nMatching {len(en_entries)} English entries with {len(vi_entries)} Vietnamese entries...")
        print(f"Mode: {self.mode} | Min Cosine: {self.min_cosine:.2f} | Min Fuzz: {self.min_fuzz}")
        print(f"Context Weight: {CONTEXT_WEIGHT} | Time Penalty: {TIME_PENALTY_FACTOR}/s | Order Penalty: {ORDER_PENALTY}")
        
        # Pre-compute contexts
        self._precompute_contexts(en_entries, vi_entries)
        
        # Pre-compute embeddings (batch) - hoặc normalized texts cho RapidFuzz
        if self.model is not None:
            self._precompute_embeddings(en_entries, vi_entries)
        else:
            # RapidFuzz fallback: lưu normalized texts
            print("  Pre-computing normalized texts for RapidFuzz...")
            for entry in en_entries:
                text = normalize_text(entry.text)
                if text:
                    self._en_normalized[entry.index] = text
            for entry in vi_entries:
                text = normalize_text(entry.text)
                if text:
                    self._vi_normalized[entry.index] = text
        
        # Detect split sentences
        split_pairs = self._detect_split_sentences(en_entries)
        if split_pairs:
            print(f"  Detected {len(split_pairs)} potential split sentence pairs")
        
        # First pass: Match tất cả
        self._last_matched_vi_idx = -1
        self._last_matched_en_start_time = -1.0
        
        temp_results = []
        
        for i, en_entry in enumerate(en_entries):
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{len(en_entries)}")
            
            match_result = self.find_best_match(
                en_entry, i, en_entries, vi_entries, used_indices
            )
            
            if match_result:
                vi_entry, score = match_result
                used_indices.add(vi_entry.index)
                temp_results.append((en_entry, vi_entry, score))
                
                # Cập nhật last matched info CHỈ KHI match đủ tin cậy (High Confidence Anchor)
                # Tính raw similarity (trước khi trừ penalties) để so sánh với threshold
                raw_similarity = self._get_similarity_score(en_entry.index, vi_entry.index)
                if raw_similarity >= HIGH_CONFIDENCE_THRESHOLD:
                    self._last_matched_vi_idx = vi_entry.index
                    self._last_matched_en_start_time = en_entry.start
            else:
                temp_results.append((en_entry, None, 0.0))
        
        # Second pass: Fix order violations trong split sentences
        temp_results = self._fix_split_sentence_order(
            temp_results, split_pairs, vi_entries
        )
        
        results = temp_results
        
        # Thống kê
        matched = sum(1 for _, vi, _ in results if vi is not None)
        print(f"\n✓ Matched: {matched}/{len(en_entries)} ({matched/len(en_entries)*100:.1f}%)")
        print(f"✗ Missing: {len(en_entries) - matched}")
        
        return results
    
    def _fix_split_sentence_order(
        self,
        results: List[Tuple[SubtitleEntry, Optional[SubtitleEntry], float]],
        split_pairs: List[Tuple[int, int]],
        vi_entries: List[SubtitleEntry]
    ) -> List[Tuple[SubtitleEntry, Optional[SubtitleEntry], float]]:
        """
        Fix các trường hợp đảo dòng trong split sentences
        
        Với mỗi cặp split (i, i+1):
        - Nếu vi_entry[i].index > vi_entry[i+1].index → đảo dòng
        - Swap lại để đảm bảo thứ tự đúng
        
        Args:
            results: Kết quả matching ban đầu
            split_pairs: Danh sách các cặp split sentences
            vi_entries: Danh sách tiếng Việt gốc
        
        Returns:
            List: Results đã fix
        """
        results = list(results)  # Copy để modify
        fixes_made = 0
        
        for idx1, idx2 in split_pairs:
            if idx1 >= len(results) or idx2 >= len(results):
                continue
            
            en1, vi1, score1 = results[idx1]
            en2, vi2, score2 = results[idx2]
            
            # Cả hai đều có match
            if vi1 is not None and vi2 is not None:
                # Check order violation: vi1 phải đứng TRƯỚC vi2
                if vi1.index > vi2.index:
                    # Swap
                    results[idx1] = (en1, vi2, score2)
                    results[idx2] = (en2, vi1, score1)
                    fixes_made += 1
        
        if fixes_made > 0:
            print(f"  Fixed {fixes_made} order violations in split sentences")
        
        return results


# ============================================================================
# EXPORT
# ============================================================================

def format_srt_time(seconds: float) -> str:
    """
    Format thời gian thành format SRT: HH:MM:SS,mmm
    
    Args:
        seconds: Thời gian (giây)
    
    Returns:
        str: Thời gian format SRT
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_ass_time(seconds: float) -> str:
    """
    Format thời gian thành format ASS: H:MM:SS.CS
    
    Args:
        seconds: Thời gian (giây)
    
    Returns:
        str: Thời gian format ASS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def export_srt(
    matches: List[Tuple[SubtitleEntry, Optional[SubtitleEntry], float]],
    output_path: str
):
    """
    Xuất file SRT
    
    Args:
        matches: Danh sách (en_entry, vi_entry, score)
        output_path: Đường dẫn file output
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (en_entry, vi_entry, score) in enumerate(matches, 1):
            # Index
            f.write(f"{i}\n")
            
            # Timing (từ tiếng Anh)
            start_time = format_srt_time(en_entry.start)
            end_time = format_srt_time(en_entry.end)
            f.write(f"{start_time} --> {end_time}\n")
            
            # Text (tiếng Việt nếu có, không thì tiếng Anh)
            if vi_entry:
                text = vi_entry.text
            else:
                text = f"[MISSING] {en_entry.text}"
            
            f.write(f"{text}\n\n")
    
    print(f"\n✓ Exported SRT: {output_path}")


def export_ass(
    matches: List[Tuple[SubtitleEntry, Optional[SubtitleEntry], float]],
    output_path: str,
    template_path: Optional[str] = None
):
    """
    Xuất file ASS
    
    Args:
        matches: Danh sách (en_entry, vi_entry, score)
        output_path: Đường dẫn file output
        template_path: Đường dẫn file ASS template (để lấy header)
    """
    # Đọc header từ template nếu có
    header = ""
    if template_path:
        with open(template_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().startswith('Dialogue:'):
                    break
                header += line
    else:
        # Header mặc định
        header = """[Script Info]
Title: Synced Vietnamese Subtitle
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Ghi header
        f.write(header)
        
        # Ghi dialogues
        for en_entry, vi_entry, score in matches:
            start_time = format_ass_time(en_entry.start)
            end_time = format_ass_time(en_entry.end)
            
            # Text (tiếng Việt nếu có, không thì tiếng Anh)
            if vi_entry:
                text = vi_entry.text
                # Thay newline thành \N cho ASS
                text = text.replace('\n', '\\N')
            else:
                text = f"[MISSING] {en_entry.text}"
                text = text.replace('\n', '\\N')
            
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")
    
    print(f"\n✓ Exported ASS: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def detect_subtitle_format(file_path: str) -> str:
    """
    Phát hiện định dạng subtitle
    
    Args:
        file_path: Đường dẫn file
    
    Returns:
        str: "srt" hoặc "ass"
    """
    # Kiểm tra đuôi file
    ext = Path(file_path).suffix.lower()
    if ext in ['.srt']:
        return 'srt'
    elif ext in ['.ass', '.ssa']:
        return 'ass'
    
    # Kiểm tra nội dung file
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            first_lines = ''.join(f.readlines(20))
            
        if '[Script Info]' in first_lines or '[Events]' in first_lines:
            return 'ass'
        elif '-->' in first_lines:
            return 'srt'
    except:
        pass
    
    raise ValueError(f"Cannot detect subtitle format for: {file_path}")


def sync_subtitles(
    en_path: str,
    vi_path: str,
    output_path: str,
    mode: str = "auto"
):
    """
    Đồng bộ subtitle tiếng Việt với timing tiếng Anh
    
    Args:
        en_path: Đường dẫn file subtitle tiếng Anh (chuẩn)
        vi_path: Đường dẫn file subtitle tiếng Việt (cần sync)
        output_path: Đường dẫn file output
        mode: Chế độ matching ("strict", "loose", "auto")
    """
    print("=" * 70)
    print("SUBTITLE SYNC TOOL v2.0")
    print("=" * 70)
    
    # Phát hiện định dạng
    print(f"\n[1] Detecting formats...")
    en_format = detect_subtitle_format(en_path)
    vi_format = detect_subtitle_format(vi_path)
    print(f"  English: {en_format.upper()}")
    print(f"  Vietnamese: {vi_format.upper()}")
    
    # Parse subtitle
    print(f"\n[2] Parsing subtitles...")
    if en_format == 'srt':
        en_entries = parse_srt(en_path)
    else:
        en_entries = parse_ass(en_path)
    print(f"  English: {len(en_entries)} entries")
    
    if vi_format == 'srt':
        vi_entries = parse_srt(vi_path)
    else:
        vi_entries = parse_ass(vi_path)
    print(f"  Vietnamese: {len(vi_entries)} entries")
    
    # Pre-splitting - Tách các dòng dài thành nhiều dòng nhỏ
    print(f"\n[3] Pre-splitting entries...")
    en_entries = apply_pre_splitting(en_entries, entry_type="EN")
    vi_entries = apply_pre_splitting(vi_entries, entry_type="VI")
    
    # Time offset detection & compensation
    print(f"\n[4] Detecting time offset...")
    offset_detector = TimeOffsetDetector()
    vi_entries_adjusted, offset_info = offset_detector.auto_detect_and_apply(
        en_entries, vi_entries, use_piecewise=True
    )
    
    print(f"  Global offset: {offset_info['global_offset']:.2f}s")
    if 'piecewise_offset' in offset_info and offset_info['piecewise_offset'] != 0:
        print(f"  Piecewise offset (first lines): {offset_info['piecewise_offset']:.2f}s")
    print(f"  Method: {offset_info['method']}")
    
    # Matching
    print(f"\n[5] Matching subtitles...")
    matcher = SubtitleMatcher(mode=mode)
    matches = matcher.match_subtitles(en_entries, vi_entries_adjusted)
    
    # Export
    print(f"\n[6] Exporting...")
    if en_format == 'srt':
        export_srt(matches, output_path)
    else:
        export_ass(matches, output_path, template_path=en_path)
    
    print("\n" + "=" * 70)
    print("✓ DONE!")
    print("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    global CONTEXT_WEIGHT, TIME_PENALTY_FACTOR, ORDER_PENALTY
    
    parser = argparse.ArgumentParser(
        description="Subtitle Sync Tool - Đồng bộ subtitle tiếng Việt với timing tiếng Anh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python subtitle_sync.py -e english.srt -v vietnamese.srt -o output.srt
  python subtitle_sync.py -e english.ass -v vietnamese.ass -o output.ass --mode loose
  
Cài đặt dependencies:
  pip install sentence-transformers rapidfuzz numpy
        """
    )
    
    parser.add_argument(
        '-e', '--english',
        required=True,
        help='File subtitle tiếng Anh (timing chuẩn) - .srt hoặc .ass'
    )
    
    parser.add_argument(
        '-v', '--vietnamese',
        required=True,
        help='File subtitle tiếng Việt (cần sync) - .srt hoặc .ass'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='File output'
    )
    
    parser.add_argument(
        '-m', '--mode',
        choices=['strict', 'loose', 'auto'],
        default='auto',
        help='Chế độ matching: strict (chặt chẽ) | loose (rộng) | auto (tự động) - default: auto'
    )
    
    parser.add_argument(
        '--context-weight',
        type=float,
        default=None,
        help=f'Trọng số context matching (0-1). Cao hơn = context quan trọng hơn. Default: {CONTEXT_WEIGHT}'
    )
    
    parser.add_argument(
        '--time-penalty',
        type=float,
        default=None,
        help=f'Time penalty factor (per second). Default: {TIME_PENALTY_FACTOR}'
    )
    
    parser.add_argument(
        '--order-penalty',
        type=float,
        default=None,
        help=f'Order violation penalty. Default: {ORDER_PENALTY}'
    )
    
    args = parser.parse_args()
    
    # Kiểm tra file tồn tại
    if not Path(args.english).exists():
        print(f"Error: English subtitle file not found: {args.english}")
        return
    
    if not Path(args.vietnamese).exists():
        print(f"Error: Vietnamese subtitle file not found: {args.vietnamese}")
        return
    
    # Apply custom config from CLI args
    if args.context_weight is not None:
        CONTEXT_WEIGHT = args.context_weight
    if args.time_penalty is not None:
        TIME_PENALTY_FACTOR = args.time_penalty
    if args.order_penalty is not None:
        ORDER_PENALTY = args.order_penalty
    
    # Chạy sync
    try:
        sync_subtitles(
            en_path=args.english,
            vi_path=args.vietnamese,
            output_path=args.output,
            mode=args.mode
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()