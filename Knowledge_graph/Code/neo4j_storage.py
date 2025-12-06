from neo4j import GraphDatabase
from pyvis.network import Network
import re
import pandas as pd
import json
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise ValueError("Neo4j credentials not found!")

def clean_text(text):
    c = re.sub(r"^\d+\.\s*", "", text.strip())
    c = re.sub(r"^\d+\s*\((.*?)\)$", r"\1", c)
    c = re.sub(r"^\d+\s*", "", c)
    c = c.strip("() ")
    return c.strip()

class Neo4jConnector:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.dataset_metadata = []  # Store dataset metadata for expansion

    def close(self):
        self.driver.close()

    def store_keywords_and_relations(self, keywords, relationships, datasets=None):
        """Store keywords, relationships, and datasets in Neo4j."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

            # Store dataset metadata for later use
            self.dataset_metadata = []
            created_datasets = []
            
            if datasets:
                print(f"\n📚 Creating {len(datasets)} dataset node(s)...")
                for dataset_info in datasets:
                    if dataset_info.get('source') != 'Not specified':
                        dataset_name = dataset_info.get('source', 'Unknown Dataset')
                        print(f"  - Creating dataset node: {dataset_name}")
                        session.execute_write(self._create_dataset_enhanced, dataset_info)
                        created_datasets.append(dataset_name)
                        self.dataset_metadata.append(dataset_info)

                        # Mark variables and link to dataset
                        for var in dataset_info.get('variables', []):
                            var_clean = clean_text(var)
                            if var_clean in [clean_text(k) for k in keywords]:
                                session.execute_write(self._mark_as_variable, var_clean)
                                session.execute_write(self._link_dataset_variable, dataset_name, var_clean)

            for key in keywords:
                cleaned_key = clean_text(key)
                print(f"Creating node: {cleaned_key}")
                session.execute_write(self._create_keyword, cleaned_key)

                # Link keyword to all datasets
                for dataset_name in created_datasets:
                    session.execute_write(self._link_dataset_keyword, dataset_name, cleaned_key)

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
    def _create_dataset_enhanced(tx, dataset_info):
        """
        Enhanced dataset creation with all metadata and PRIMARY/CITED labels.

        Creates nodes with dual labels for easy querying:
        - All datasets: :Dataset label
        - PRIMARY datasets: :Dataset:PrimaryDataset labels
        - CITED datasets: :Dataset:CitedDataset labels

        This allows queries like:
        - MATCH (d:Dataset) -> all datasets
        - MATCH (d:PrimaryDataset) -> only PRIMARY
        - MATCH (d:CitedDataset) -> only CITED
        """
        dataset_type = dataset_info.get('dataset_type', 'cited')

        # Determine type-specific label (PLUGGABLE: easy to add more types)
        if dataset_type == 'primary':
            type_label = 'PrimaryDataset'
        elif dataset_type == 'cited':
            type_label = 'CitedDataset'
        else:
            type_label = 'UnknownDataset'  # Fallback for unexpected types

        # Build query with dual labels: Dataset + type-specific label
        query = f"""
        MERGE (d:Dataset:{type_label} {{
            name: $name,
            time_period: $time_period,
            location: $location,
            dataset_type: $dataset_type,
            usage_description: $usage_description,
            citation_info: $citation_info,
            confidence: $confidence,
            context: $context
        }})
        """

        # Execute with all parameters (defensive: use .get() with defaults, handle None)
        tx.run(query,
               name=dataset_info.get('source') or 'Unknown',
               time_period=dataset_info.get('time_period') or 'Not specified',
               location=dataset_info.get('location') or 'Not specified',
               dataset_type=dataset_type,
               usage_description=dataset_info.get('usage_description') or '',
               citation_info=dataset_info.get('citation_info') or '',
               confidence=float(dataset_info.get('confidence') or 0.5),
               context=dataset_info.get('context') or '')
    
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

    def get_datasets_by_type(self, dataset_type='all'):
        """
        Retrieve datasets filtered by type (PLUGGABLE: easy filtering).

        Args:
            dataset_type: 'all', 'primary', or 'cited'

        Returns:
            List of dataset dictionaries with all properties

        Examples:
            >>> connector.get_datasets_by_type('primary')  # Only PRIMARY
            >>> connector.get_datasets_by_type('cited')    # Only CITED
            >>> connector.get_datasets_by_type('all')      # All datasets
        """
        # Build query based on type (defensive: handle invalid types)
        if dataset_type == 'primary':
            query = "MATCH (d:PrimaryDataset) RETURN d"
        elif dataset_type == 'cited':
            query = "MATCH (d:CitedDataset) RETURN d"
        elif dataset_type == 'all':
            query = "MATCH (d:Dataset) RETURN d"
        else:
            # Fallback for invalid type
            print(f"⚠️  Invalid dataset_type: {dataset_type}. Using 'all'.")
            query = "MATCH (d:Dataset) RETURN d"

        with self.driver.session() as session:
            result = session.run(query)
            datasets = []
            for record in result:
                # Extract all properties from the node
                node = record['d']
                datasets.append(dict(node))
            return datasets

    def generate_graph(self, relations, graph_type='with_datasets'):
        """
        Generate graph based on visualization type.

        Args:
            relations: List of relationship dictionaries
            graph_type: Type of graph to generate
                - 'with_datasets': Hub-based with collapsible datasets (default)
                - 'without_datasets': Knowledge graph only (no datasets)

        Returns:
            tuple: (network, expansion_js)
        """
        net = Network(height="600px", width="100%", directed=True)

        # Get node types from database
        with self.driver.session() as session:
            dataset_query = """
            MATCH (d:Dataset)
            RETURN d.name as name,
                   d.dataset_type as dataset_type,
                   d.time_period as time_period,
                   d.location as location,
                   d.usage_description as usage_description,
                   d.confidence as confidence
            """
            # Always load dataset data if graph_type is with_datasets
            dataset_data = session.run(dataset_query).data() if graph_type == 'with_datasets' else []
            variable_nodes = session.run("MATCH (v:Variable) RETURN v.name as name").data()

        dataset_info = {d['name']: d for d in dataset_data}
        variable_names = {v['name'] for v in variable_nodes}

        added_nodes = set()

        # For 'with_datasets': Create hub node
        if graph_type == 'with_datasets' and dataset_data:
            primary_count = sum(1 for d in dataset_data if d.get('dataset_type') == 'primary')
            cited_count = len(dataset_data) - primary_count
            hub_tooltip = f"📊 Datasets Hub\n\n🟩 PRIMARY: {primary_count}\n🔵 CITED: {cited_count}\n\n🖱️ Double-click to expand"
            net.add_node(
                "📊 Datasets",
                label="📊 Datasets",
                color='#9c27b0',  # Purple for hub
                shape='box',
                size=35,
                title=hub_tooltip,
                is_dataset_hub=True
            )
            added_nodes.add("📊 Datasets")

        # Process relations (regular nodes)
        for r in relations:
            src = r["source"]
            tgt = r["target"]
            rel = r["relation"]

            # Skip dataset nodes in relations for both modes
            # - 'with_datasets': datasets are hidden (only hub visible)
            # - 'without_datasets': datasets are excluded entirely
            if src in dataset_info or tgt in dataset_info:
                continue

            # Add source node
            if src not in added_nodes:
                if src in variable_names:
                    net.add_node(src, label=src, color='#fbbc04', title=f"Variable: {src}")
                else:
                    net.add_node(src, label=src)
                added_nodes.add(src)

            # Add target node
            if tgt not in added_nodes:
                if tgt in variable_names:
                    net.add_node(tgt, label=tgt, color='#fbbc04', title=f"Variable: {tgt}")
                else:
                    net.add_node(tgt, label=tgt)
                added_nodes.add(tgt)

            # Add edge
            net.add_edge(src, tgt, label=rel)

        # Connect hub to a central keyword (only for 'with_datasets')
        if graph_type == 'with_datasets' and dataset_data and added_nodes:
            # Find a central keyword to connect to (exclude the hub itself)
            keyword_nodes = [n for n in added_nodes if n != "📊 Datasets"]
            if keyword_nodes:
                central_keyword = keyword_nodes[0]
                net.add_edge("📊 Datasets", central_keyword, label="CONTAINS", color='#9c27b0', dashes=True)

        net.repulsion(node_distance=200, spring_length=300)

        # Generate expansion JavaScript only for 'with_datasets'
        if graph_type == 'with_datasets' and dataset_info:
            expansion_js = self._generate_expansion_javascript(dataset_info)
        else:
            expansion_js = ""

        return net, expansion_js

    def _generate_expansion_javascript(self, dataset_info):
        """Generate JavaScript code for expanding dataset hub on double-click."""
        
        # Prepare dataset data for JavaScript
        primary_datasets = []
        cited_datasets = []
        
        for name, info in dataset_info.items():
            dataset_obj = {
                'id': name,
                'label': name,
                'type': info.get('dataset_type', 'unknown'),
                'time': info.get('time_period', 'Not specified'),
                'location': info.get('location', 'Not specified'),
                'usage': info.get('usage_description', '')[:80] + '...' if info.get('usage_description') else '',
                'confidence': info.get('confidence', 0.0)
            }
            
            if info.get('dataset_type') == 'primary':
                primary_datasets.append(dataset_obj)
            else:
                cited_datasets.append(dataset_obj)
        
        datasets_json = json.dumps({
            'primary': primary_datasets,
            'cited': cited_datasets
        })
        
        js_code = f"""
        <script type="text/javascript">
        var datasetsExpanded = false;
        var datasetNodes = {datasets_json};
        var addedDatasetIds = [];
        
        network.on("doubleClick", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                
                // Check if this is the Datasets hub
                if (nodeId === "📊 Datasets") {{
                    if (!datasetsExpanded) {{
                        // Expand: Add all dataset nodes
                        var newNodes = [];
                        var newEdges = [];
                        
                        // Add PRIMARY datasets (green)
                        datasetNodes.primary.forEach(function(ds, idx) {{
                            var nodeId = 'primary_' + idx;
                            var tooltip = '🟩 PRIMARY Dataset: ' + ds.label + '\\n';
                            if (ds.time !== 'Not specified') tooltip += '⏰ ' + ds.time + '\\n';
                            if (ds.location !== 'Not specified') tooltip += '📍 ' + ds.location + '\\n';
                            if (ds.usage) tooltip += '💡 ' + ds.usage + '\\n';
                            tooltip += '✨ Confidence: ' + ds.confidence.toFixed(2);
                            
                            newNodes.push({{
                                id: nodeId,
                                label: ds.label,
                                color: '#34a853',  // Green
                                shape: 'box',
                                size: 25,
                                title: tooltip,
                                x: params.pointer.canvas.x - 200 + (idx * 50),
                                y: params.pointer.canvas.y - 100
                            }});
                            
                            newEdges.push({{
                                from: "📊 Datasets",
                                to: nodeId,
                                color: '#34a853',
                                dashes: true
                            }});
                            
                            addedDatasetIds.push(nodeId);
                        }});
                        
                        // Add CITED datasets (blue)
                        datasetNodes.cited.forEach(function(ds, idx) {{
                            var nodeId = 'cited_' + idx;
                            var tooltip = '🔵 CITED Dataset: ' + ds.label + '\\n';
                            if (ds.time !== 'Not specified') tooltip += '⏰ ' + ds.time + '\\n';
                            if (ds.location !== 'Not specified') tooltip += '📍 ' + ds.location + '\\n';
                            if (ds.usage) tooltip += '💡 ' + ds.usage + '\\n';
                            tooltip += '✨ Confidence: ' + ds.confidence.toFixed(2);
                            
                            newNodes.push({{
                                id: nodeId,
                                label: ds.label,
                                color: '#4285f4',  // Blue
                                shape: 'box',
                                size: 25,
                                title: tooltip,
                                x: params.pointer.canvas.x - 200 + (idx * 50),
                                y: params.pointer.canvas.y + 100
                            }});
                            
                            newEdges.push({{
                                from: "📊 Datasets",
                                to: nodeId,
                                color: '#4285f4',
                                dashes: true
                            }});
                            
                            addedDatasetIds.push(nodeId);
                        }});
                        
                        // Add nodes and edges
                        nodes.add(newNodes);
                        edges.add(newEdges);
                        
                        datasetsExpanded = true;
                        
                    }} else {{
                        // Collapse: Remove all dataset nodes
                        nodes.remove(addedDatasetIds);
                        addedDatasetIds = [];
                        datasetsExpanded = false;
                    }}
                }}
            }}
        }});
        </script>
        """
        
        return js_code

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
