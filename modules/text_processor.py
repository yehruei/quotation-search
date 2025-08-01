import pypdf
import re

def process_pdf(file_path, chunk_size=800, chunk_overlap=80, remove_stopwords=False):
    """
    Extracts text from a PDF file, page by page, with configurable text processing options.
    
    Args:
        file_path: Path to the PDF file
        chunk_size: Maximum size of text chunks (in characters)
        chunk_overlap: Overlap between consecutive chunks (in characters)
        remove_stopwords: Whether to remove stopwords from the text
    
    Returns:
        List of processed text chunks
    """
    texts = []
    try:
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    # Clean and preprocess text
                    cleaned_text = clean_text(text, remove_stopwords)
                    
                    # Split into chunks if text is too long
                    if len(cleaned_text) > chunk_size:
                        chunks = split_text_into_chunks(cleaned_text, chunk_size, chunk_overlap)
                        texts.extend(chunks)
                    else:
                        texts.append(cleaned_text)
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return texts

def clean_text(text, remove_stopwords=False):
    """
    Clean and preprocess text.
    
    Args:
        text: Raw text to clean
        remove_stopwords: Whether to remove stopwords
    
    Returns:
        Cleaned text
    """
    # Remove PDF format tags and markers
    text = remove_pdf_format_tags(text)
    
    # Basic text cleaning
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = text.strip()
    
    if remove_stopwords:
        text = remove_stopwords_from_text(text)
    
    return text

def remove_pdf_format_tags(text):
    """
    Remove PDF format tags and markers from text.
    
    Args:
        text: Text containing PDF format tags
    
    Returns:
        Cleaned text without PDF format tags
    """
    # Remove FTLN (First Through Line Number) tags and similar PDF formatting
    text = re.sub(r'\bFTLN\s*\d+\b', '', text)  # Remove FTLN followed by numbers
    text = re.sub(r'\bTLN\s*\d+\b', '', text)   # Remove TLN followed by numbers
    text = re.sub(r'\bLN\s*\d+\b', '', text)    # Remove LN followed by numbers
    
    # Remove page numbers and headers/footers patterns
    text = re.sub(r'\b(?:Page|页码|第.*页)\s*\d+\b', '', text, flags=re.IGNORECASE)
    
    # Remove common PDF metadata tags
    text = re.sub(r'\b(?:PDF|Adobe|Creator|Producer)\b.*?\n', '', text, flags=re.IGNORECASE)
    
    # Remove line numbers at start of lines (pattern: number followed by space/tab)
    text = re.sub(r'^\s*\d+\s+', '', text, flags=re.MULTILINE)
    
    # Remove timestamp patterns
    text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text)
    
    # Remove reference markers like [1], (1), etc.
    text = re.sub(r'[\[\(]\d+[\]\)]', '', text)
    
    # Remove extra whitespace created by removals
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def remove_stopwords_from_text(text):
    """
    Remove common stopwords from text.
    
    Args:
        text: Text to process
    
    Returns:
        Text with stopwords removed
    """
    # Common Chinese and English stopwords
    stopwords = {
        # English stopwords
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 
        'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 'they', 'we', 
        'she', 'he', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        
        # Chinese stopwords  
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', 
        '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那'
    }
    
    # Split text into words and filter out stopwords
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    
    return ' '.join(filtered_words)

def split_text_into_chunks(text, chunk_size, chunk_overlap):
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this is the last chunk, take all remaining text
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at a sentence boundary if possible
        chunk_text = text[start:end]
        
        # Look for sentence endings near the chunk boundary
        sentence_endings = ['.', '!', '?', '。', '！', '？']
        best_break = end
        
        # Look backward from the end for a good break point
        for i in range(min(100, len(chunk_text))):  # Look back up to 100 chars
            pos = end - 1 - i
            if pos <= start:
                break
            if text[pos] in sentence_endings and pos > start + chunk_size // 2:
                best_break = pos + 1
                break
        
        chunks.append(text[start:best_break])
        start = best_break - chunk_overlap
        
        # Ensure we don't go backwards
        if start <= chunks[-1].__len__() - chunk_size:
            start = best_break
    
    return [chunk.strip() for chunk in chunks if chunk.strip()]