import pdfplumber

pdf_path = "NCERT_Class10_Science.pdf"

with pdfplumber.open(pdf_path) as pdf:
    all_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

print(all_text[:1000])  # See the first 1000 characters
