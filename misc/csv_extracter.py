import os
import csv

ROOT_FOLDER = r"C:\Users\adrie\Documents\Projects\ScribbleNet\cvl-database-1-1"

OUTPUT_CSV = "cvl_words_dataset_detailed.csv"

rows = []

for root, dirs, files in os.walk(ROOT_FOLDER):

    if "words" not in root.lower():
        continue

    if any(skip in root.lower() for skip in ["lines", "pages", "xml"]):
        continue

    for file in files:
        if file.lower().endswith(".tif"):

            full_path = os.path.join(root, file)
            filename = file

            name_without_ext = os.path.splitext(file)[0]
            parts = name_without_ext.split("-")

            if len(parts) >= 5:
                writer_id = parts[0]
                text_id = parts[1]
                line_id = parts[2]
                word_index = parts[3]
                label = "-".join(parts[4:])  # handles hyphenated words
            else:
                continue

            rows.append([
                full_path,
                filename,
                writer_id,
                text_id,
                line_id,
                word_index,
                label
            ])

with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "full_path",
        "filename",
        "writer_id",
        "text_id",
        "line_id",
        "word_index",
        "label"
    ])
    writer.writerows(rows)

print(f"✅ Detailed CSV created with {len(rows)} samples!")