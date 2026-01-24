# utils/pdf_utils.py
import os
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF using pdfplumber only."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"PDF extraction error for {pdf_path}: {e}")

    return text


def extract_text_from_folder(folder_path):
    all_texts = {}
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return all_texts

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Extracting: {pdf_file}")
        text = extract_text_from_pdf(pdf_path)
        all_texts[pdf_file] = text

    return all_texts
