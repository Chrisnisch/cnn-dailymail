import json
import torch

def prepare_custom_text(text):
    """CNN/DailyMail"""
    
    # ������� �����������
    sentences = text.split('. ')
    tokens = []
    for sent in sentences:
        tokens.extend(sent.lower().split())
    
    # ������� ��������� ��� � CNN/DailyMail
    data = {
        "article": tokens,
        "abstract": [],  # ������ ������ ��� ���������
        "id": "custom_001"
    }
    
    # ��������� � �������, ������� ������� ������
    with open('custom_article.json', 'w') as f:
        json.dump(data, f)
    
    return data

# ��� �����
my_text = """
    text in these quotes will be tokenized
"""

prepare_custom_text(my_text)