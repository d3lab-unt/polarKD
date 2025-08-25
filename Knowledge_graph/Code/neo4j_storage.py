from neo4j import GraphDatabase
from pyvis.network import Network
import re
import pandas as pd
import json

# Neo4j credentials - Updated to new instance
NEO4J_URI='neo4j+s://0d4ad98d.databases.neo4j.io'
NEO4J_USER='neo4j'
NEO4J_PASSWORD='l2eTsa3JmSPkwWoCCNszhUyvkxkapl3WwN2oHzJZJ6E'

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

    def store_keywords_and_relations(self, keywords, relationships, dataset_info=None):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create dataset node if dataset info is provided
            if dataset_info and dataset_info.get('source') != 'Not specified':
                dataset_name = dataset_info.get('source', 'Unknown Dataset')
                print(f"Creating dataset node: {dataset_name}")
                session.execute_write(self._create_dataset, dataset_info)
                
                # Mark variables as special keywords and link to dataset
                for var in dataset_info.get('variables', []):
                    var_clean = clean_text(var)
                    if var_clean in [clean_text(k) for k in keywords]:
                        session.execute_write(self._mark_as_variable, var_clean)
                        session.execute_write(self._link_dataset_variable, dataset_name, var_clean)
            
            for key in keywords:
                cleaned_key = clean_text(key)
                print(f"Creating node: {cleaned_key}")
                session.execute_write(self._create_keyword, cleaned_key)
                
                # Link keyword to dataset if dataset exists
                if dataset_info and dataset_info.get('source') != 'Not specified':
                    session.execute_write(self._link_dataset_keyword, dataset_info.get('source'), cleaned_key)

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
    def _create_dataset(tx, dataset_info):
        query = """
        MERGE (d:Dataset {
            name: $name, 
            time_period: $time_period,
            location: $location
        })
        """
        tx.run(query, 
               name=dataset_info.get('source', 'Unknown'),
               time_period=dataset_info.get('time_period', 'Not specified'),
               location=dataset_info.get('location', 'Not specified'))
    
    @staticmethod
    def _mark_as_variable(tx, keyword):
        query = "MATCH (k:Keyword {name: $keyword}) SET k:Variable"
        tx.run(query, keyword=keyword)
    
    @staticmethod
    def _link_dataset_variable(tx, dataset_name, variable):
        query = """
        MATCH (d:Dataset {name: $dataset})
        MATCH (v:Keyword {name: $variable})
        MERGE (d)-[:HAS_VARIABLE]->(v)
        """
        tx.run(query, dataset=dataset_name, variable=variable)
    
    @staticmethod
    def _link_dataset_keyword(tx, dataset_name, keyword):
        query = """
        MATCH (d:Dataset {name: $dataset})
        MATCH (k:Keyword {name: $keyword})
        MERGE (d)-[:EXTRACTED_FROM]->(k)
        """
        tx.run(query, dataset=dataset_name, keyword=keyword)
    
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
        
        # Get node types from database
        with self.driver.session() as session:
            dataset_nodes = session.run("MATCH (d:Dataset) RETURN d.name as name").data()
            variable_nodes = session.run("MATCH (v:Variable) RETURN v.name as name").data()
        
        dataset_names = {d['name'] for d in dataset_nodes}
        variable_names = {v['name'] for v in variable_nodes}
        
        added_nodes = set()
        
        for r in relations:
            src = r["source"]
            tgt = r["target"]
            rel = r["relation"]
            
            # Add source node with appropriate styling
            if src not in added_nodes:
                if src in dataset_names:
                    # Dataset node - blue square, larger
                    net.add_node(src, label=src, color='#4285f4', shape='square', size=25, 
                               title=f"Dataset: {src}")
                elif src in variable_names:
                    # Variable node - yellow circle
                    net.add_node(src, label=src, color='#fbbc04', title=f"Variable: {src}")
                else:
                    # Regular keyword - default styling
                    net.add_node(src, label=src)
                added_nodes.add(src)
            
            # Add target node with appropriate styling
            if tgt not in added_nodes:
                if tgt in dataset_names:
                    # Dataset node - blue square, larger
                    net.add_node(tgt, label=tgt, color='#4285f4', shape='square', size=25,
                               title=f"Dataset: {tgt}")
                elif tgt in variable_names:
                    # Variable node - yellow circle
                    net.add_node(tgt, label=tgt, color='#fbbc04', title=f"Variable: {tgt}")
                else:
                    # Regular keyword - default styling
                    net.add_node(tgt, label=tgt)
                added_nodes.add(tgt)
            
            # Add edge with different style for dataset connections
            if rel in ['EXTRACTED_FROM', 'HAS_VARIABLE']:
                net.add_edge(src, tgt, label=rel, color='#4285f4', dashes=True)
            else:
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