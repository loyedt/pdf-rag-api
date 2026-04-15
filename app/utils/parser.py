from pypdf import PdfReader

def parse_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for i, page in enumerate(reader.pages):
        text += page.extract_text() + "\n"

    return text