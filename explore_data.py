import os
import re
import json

# You need to install pdfplumber before running: pip install pdfplumber
import pdfplumber

def clean_text(text):
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:\'\"()\-\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=250):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = words[start:end]
        chunks.append(' '.join(chunk))
        start += (chunk_size - overlap)
    return chunks

def extract_text_from_pdf(pdf_path):
    text = ''
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def preprocess_all_contracts(txt_folder, pdf_folder, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        
        # Process TXT files
        for filename in os.listdir(txt_folder):
            if filename.endswith('.txt'):
                contract_id = os.path.splitext(filename)[0]
                txt_path = os.path.join(txt_folder, filename)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                cleaned = clean_text(raw_text)
                chunks = chunk_text(cleaned)
                for i, chunk in enumerate(chunks):
                    json_line = json.dumps({
                        "contract_id": contract_id,
                        "chunk_id": f"{contract_id}_chunk_{i+1:03d}",
                        "text": chunk
                    })
                    out_f.write(json_line + "\n")

        # Process PDF files
        for filename in os.listdir(pdf_folder):
            if filename.endswith('.pdf'):
                contract_id = os.path.splitext(filename)[0]
                pdf_path = os.path.join(pdf_folder, filename)
                raw_text = extract_text_from_pdf(pdf_path)
                if not raw_text.strip():
                    print(f"No text extracted from {filename}, skipping.")
                    continue
                cleaned = clean_text(raw_text)
                chunks = chunk_text(cleaned)
                for i, chunk in enumerate(chunks):
                    json_line = json.dumps({
                        "contract_id": contract_id,
                        "chunk_id": f"{contract_id}_chunk_{i+1:03d}",
                        "text": chunk
                    })
                    out_f.write(json_line + "\n")

    print(f"Preprocessing complete. Output saved to {output_file}")

if __name__ == "__main__":
    txt_folder = './full_contract_txt'  # Adjust to your txt folder path
    pdf_folder = './full_contract_pdf'  # Adjust to your pdf folder path
    output_file = 'preprocessed_chunks.jsonl'
    preprocess_all_contracts(txt_folder, pdf_folder, output_file)
