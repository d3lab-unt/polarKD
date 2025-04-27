from neo4j import GraphDatabase
from pyvis.network import Network
import re
import pandas as pd
import json

# Neo4j credentials
NEO4J_URI='neo4j+s://abb45abd.databases.neo4j.io'
NEO4J_USER='neo4j'
NEO4J_PASSWORD='L79SPg4lF0BFy37Nf4VWjeEocRVyjAW_UHtpnzu5ZG4'

def clean_text(text):
    c = re.sub(r"^\d+\.\s*", "", text.strip())
    c = re.sub(r"^\d+\s*\((.*?)\)$", r"\1", c)
    c = re.sub(r"^\d+\s*", "", c)
    c = c.strip("() ")
    return c.strip()

class Neo4jConnector:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def store_keywords_and_relations(self, keywords, relationships):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            for key in keywords:
                cleaned_key = clean_text(key)
                print(f"Creating node: {cleaned_key}")
                session.execute_write(self._create_keyword, cleaned_key)

            for rel_dict in relationships:
                k1 = rel_dict['source']
                rel = rel_dict['relation']
                k2 = rel_dict['target']
                cleaned_k1 = clean_text(k1)
                cleaned_k2 = clean_text(k2)
                rel = rel.upper().replace(" ", "_").replace("-", "_").strip()
                if not re.match(r"^[A-Z_][A-Z0-9_]*$", rel):
                    print(f"Invalid relation name, so skipping them: {rel}")
                    continue
                print(f"Creating relation: ({cleaned_k1}) -[{rel}]-> ({cleaned_k2})")
                session.execute_write(self._create_relationship, cleaned_k1, rel, cleaned_k2)
        print("\n Keywords and relations stored in Neo4j!")

    @staticmethod
    def _create_keyword(tx, keyword):
        query = "MERGE (k:Keyword {name: $keyword})"
        tx.run(query, keyword=keyword)

    @staticmethod
    def _create_relationship(tx, k1, rel, k2):
        query = f"""
        MATCH (a:Keyword {{name: $k1}}), (b:Keyword {{name: $k2}})
        MERGE (a)-[r:{rel}]->(b)
        """
        tx.run(query, k1=k1, k2=k2)

    def retrieve_relations(self):
        query = """
        MATCH (a:Keyword)-[r]->(b:Keyword)
        RETURN a.name AS source, type(r) AS relation, b.name AS target
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [
                {"source": r["source"], "relation": r["relation"], "target": r["target"]}
                for r in result
            ]

    def generate_graph(self, relations):
        net = Network(height="600px", width="100%", directed=True)
        for r in relations:
            src = r["source"]
            tgt = r["target"]
            rel = r["relation"]
            net.add_node(src, label=src)
            net.add_node(tgt, label=tgt)
            net.add_edge(src, tgt, label=rel)
        net.repulsion(node_distance=200, spring_length=300)
        return net

    def export_csv(self, filepath='extracted_relations.csv'):
        r = self.retrieve_relations()
        data = pd.DataFrame(r)
        data.to_csv(filepath, index=False)
        print(f"✅ The relations are stored in {filepath}")
        return data

    def export_json(self, filepath="extracted_relations.json"):
        r = self.retrieve_relations()
        with open(filepath, "w") as f:
            json.dump(r, f, indent=2)
        print(f"✅ The relations are stored in {filepath}")
        return r