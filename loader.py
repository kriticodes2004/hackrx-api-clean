import os
import requests
import tempfile
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from docx import Document

def load_document_from_url(url: str) -> str:
   
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    filename = url.split("/")[-1].split("?")[0]
    ext = os.path.splitext(filename)[-1].lower()

    
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp_file.write(response.content)
        tmp_file.close()

        text = extract_text(tmp_file.name)
    finally:
     
        if os.path.exists(tmp_file.name):
            os.remove(tmp_file.name)

    return text.strip()


def load_document_from_path(file_path: str) -> str:
   
    return extract_text(file_path)


def extract_text(file_path: str) -> str:
    
    ext = os.path.splitext(file_path)[-1].lower()
    text = ""

    if ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

    elif ext in [".html", ".htm"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator="\n")

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    elif ext == ".docx":
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])

    else:
        raise Exception(f"Unsupported file format: {ext}")

    return text.strip()


if __name__ == "__main__":
    test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    print(" Extracted Text Preview:\n", load_document_from_url(test_url)[:500])
