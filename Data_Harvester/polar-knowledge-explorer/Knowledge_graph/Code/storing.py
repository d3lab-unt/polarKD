# storing.py
from keywords_extraction import process  # Your processing function
from neo4j_storage import Neo4jConnector  # Your Neo4j connector class
from collections import defaultdict
import pandas as pd
import sys
# Dictionary storage for multiple relations between same pairs
def dictionary_relations(relations):
    r = defaultdict(list)
    for key1, relation, key2, _ in relations:
        r[(key1, key2)].append(relation)
        r[(key2, key1)].append(relation)
    return r

# PDF input
pdf_path = sys.argv[1] if len(sys.argv) > 1 else "temp.pdf"
# Step 1: Extract keywords and relations
nodes, relations = process(pdf_path, k=15)

# Step 2: Print for verification
print("Extracted Keywords:")
print(nodes)
print("\nExtracted Relations:")
for rel in relations:
    print(rel)

# Step 3: Create dictionary
rel_dict = dictionary_relations(relations)
print("\nDictionary of keyword pairs and relations:")
for i, j in rel_dict.items():
    print(f"{i} ---> {j}")

# Step 4: Store and visualize
neo = Neo4jConnector()
neo.store_keywords_and_relations(nodes, relations)
rels = neo.retrieve_relations()

# Save CSV and JSON
csv_df = neo.export_csv(filepath='extracted_relations.csv')
json_data = neo.export_json(filepath='extracted_relations.json')

# Generate and display knowledge graph
graph = neo.generate_graph(rels)
graph.show("graph.html")

# Close connection
neo.close()
print("\n\U0001F680 Keywords and relations stored & visualized successfully!")