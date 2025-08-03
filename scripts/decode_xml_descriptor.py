import os
import base64
import xml.etree.ElementTree as ET
from pathlib import Path

# Directory for saving extracted data
xml_file_path = "data/Angebotsbeschreibung.xml"
output_folder = "extracted_docx"
Path(output_folder).mkdir(parents=True, exist_ok=True)

def extract_text_and_images(xml_file_path):
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Define namespaces
    namespaces = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'pkg': 'http://schemas.microsoft.com/office/2006/xmlPackage',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    }

    # Extract text content
    text_output = []
    for paragraph in root.findall(".//w:p", namespaces):
        for run in paragraph.findall(".//w:t", namespaces):
            text = run.text
            if text:
                text_output.append(text.strip())

    # Save extracted text
    with open(os.path.join(output_folder, "extracted_text.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(text_output))
    print(f"Text saved to {os.path.join(output_folder, 'extracted_text.txt')}")

    # Extract image data
    for part in root.findall(".//pkg:part[@pkg:name='/word/media/image1.png']", namespaces):
        binary_data = part.find("pkg:binaryData", namespaces).text
        decoded_data = base64.b64decode(binary_data)
        image_path = os.path.join(output_folder, "image1.png")
        with open(image_path, "wb") as img_file:
            img_file.write(decoded_data)
        print(f"Image saved to {image_path}")

    # Extract metadata
    metadata = {}
    core_props = root.find(".//pkg:part[@pkg:name='/docProps/core.xml']/pkg:xmlData/cp:coreProperties", namespaces)
    if core_props is not None:
        metadata['creator'] = core_props.find("dc:creator", namespaces).text
        metadata['created'] = core_props.find("dcterms:created", namespaces).text
        metadata['modified'] = core_props.find("dcterms:modified", namespaces).text
    app_props = root.find(".//pkg:part[@pkg:name='/docProps/app.xml']/pkg:xmlData/Properties", namespaces)
    if app_props is not None:
        metadata['words'] = app_props.find("Words", namespaces).text
        metadata['pages'] = app_props.find("Pages", namespaces).text
        metadata['characters'] = app_props.find("Characters", namespaces).text

    # Save metadata
    with open(os.path.join(output_folder, "metadata.txt"), "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"Metadata saved to {os.path.join(output_folder, 'metadata.txt')}")

# Execute the script
if __name__ == "__main__":
    if not os.path.exists(xml_file_path):
        print(f"File {xml_file_path} not found.")
    else:
        extract_text_and_images(xml_file_path)