import csv
import json

def csv_to_json(csv_file_path, json_file_path):

    data = []

    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for idx, row in enumerate(reader, start=1):
            
            new_entry = {
                "id": f"DATA{idx:05d}",  
                "ec": row.get('EC', ''),
                "type": row.get('EnzymeType', ''),
                "organism": row.get('Organism', ''),
                "sequence": row.get('Sequence', ''),
                "substrate": row.get('Substrate', ''),
                "smiles": row.get('Smiles', ''),
                "value": float(row.get('kcat/km()', 0)) if row.get('kcat/km()') else 0.0,
                "source": "custom", 
                "uniprotID": row.get('UniProtID', '')
            }
            data.append(new_entry)
    
    
    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False)
    
    print(f" {len(data)} {json_file_path}")


csv_to_json('merged_output.csv', 'merged_output.json')