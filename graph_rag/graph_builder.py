import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


class GraphBuilder:
    def __init__(self, batch_size=3):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found in the .env file."
            )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key,
            temperature=0
        )

        self.batch_size = batch_size
        self.nodes = {}
        self.edges = []

        print("Graph RAG graph builder initialized successfully!")

    def extract_text(self, response):
        if isinstance(response.content, str):
            return response.content.strip()

        return "\n".join(
            item["text"]
            for item in response.content
            if isinstance(item, dict)
            and item.get("type") == "text"
        ).strip()

    def safe_json_parse(self, text):
        try:
            start = text.find("{")
            end = text.rfind("}")

            if start == -1 or end == -1:
                return {
                    "entities": [],
                    "relationships": []
                }

            json_text = text[start:end + 1]

            data = json.loads(json_text)

            if not isinstance(data, dict):
                return {
                    "entities": [],
                    "relationships": []
                }

            return {
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", [])
            }

        except Exception:
            print("Warning: Could not parse graph extraction response.")

            return {
                "entities": [],
                "relationships": []
            }

    def extract_graph_from_batch(self, documents):
        combined_text = ""

        for index, document in enumerate(documents, start=1):
            combined_text += (
                f"\n\n--- DOCUMENT CHUNK {index} ---\n"
                f"{document.page_content}"
            )

        prompt = f"""
You are a knowledge graph extraction system.

Extract important entities and relationships
from the document chunks below.

{combined_text}

Return ONLY valid JSON in exactly this format:

{{
    "entities": [
        {{
            "name": "entity name",
            "type": "entity type"
        }}
    ],
    "relationships": [
        {{
            "source": "source entity",
            "relationship": "relationship",
            "target": "target entity"
        }}
    ]
}}

Rules:

1. Extract only information explicitly present
   in the document chunks.

2. Do not invent entities.

3. Do not invent relationships.

4. Keep entity names concise.

5. Keep relationship names concise.

6. Use the exact same entity names in
   relationships as in the entities list.

7. Combine duplicate entities.

8. Combine duplicate relationships.

9. Return JSON only.

JSON:
"""

        try:
            response = self.llm.invoke(prompt)

            text_response = self.extract_text(response)

            return self.safe_json_parse(text_response)

        except Exception as error:
            print("\nGraph extraction failed.")
            print(f"Reason: {error}")

            return {
                "entities": [],
                "relationships": []
            }

    def add_entity(self, name, entity_type):
        if not isinstance(name, str):
            return

        normalized_name = name.strip()

        if not normalized_name:
            return

        key = normalized_name.lower()

        if key not in self.nodes:
            self.nodes[key] = {
                "name": normalized_name,
                "type": entity_type or "Unknown"
            }

    def add_relationship(self, source, relationship, target):
        if not isinstance(source, str):
            return

        if not isinstance(relationship, str):
            return

        if not isinstance(target, str):
            return

        source = source.strip()
        relationship = relationship.strip()
        target = target.strip()

        if not source or not relationship or not target:
            return

        edge = {
            "source": source,
            "relationship": relationship,
            "target": target
        }

        for existing_edge in self.edges:

            same_source = (
                existing_edge["source"].lower()
                == source.lower()
            )

            same_relationship = (
                existing_edge["relationship"].lower()
                == relationship.lower()
            )

            same_target = (
                existing_edge["target"].lower()
                == target.lower()
            )

            if (
                same_source
                and same_relationship
                and same_target
            ):
                return

        self.edges.append(edge)

    def build_graph(self, documents):
        print("\n")
        print("=" * 70)
        print("BUILDING KNOWLEDGE GRAPH")
        print("=" * 70)

        print(f"\nProcessing {len(documents)} chunks...")

        total_batches = (
            len(documents) + self.batch_size - 1
        ) // self.batch_size

        for batch_number in range(total_batches):

            start = batch_number * self.batch_size

            end = start + self.batch_size

            batch = documents[start:end]

            print(
                f"\nProcessing batch "
                f"{batch_number + 1}/{total_batches}..."
            )

            graph_data = self.extract_graph_from_batch(batch)

            entities = graph_data.get(
                "entities",
                []
            )

            relationships = graph_data.get(
                "relationships",
                []
            )

            for entity in entities:

                if not isinstance(entity, dict):
                    continue

                self.add_entity(
                    entity.get("name", ""),
                    entity.get("type", "Unknown")
                )

            for relationship in relationships:

                if not isinstance(relationship, dict):
                    continue

                self.add_relationship(
                    relationship.get("source", ""),
                    relationship.get("relationship", ""),
                    relationship.get("target", "")
                )

        print("\n")
        print("=" * 70)
        print("KNOWLEDGE GRAPH CREATED")
        print("=" * 70)

        print(
            f"\nNumber of nodes: {len(self.nodes)}"
        )

        print(
            f"Number of relationships: {len(self.edges)}"
        )

        return self.get_graph()

    def get_graph(self):
        return {
            "nodes": self.nodes,
            "edges": self.edges
        }

    def print_graph(self):
        print("\n")
        print("=" * 70)
        print("KNOWLEDGE GRAPH")
        print("=" * 70)

        print("\nEntities:")

        for node in self.nodes.values():
            print(
                f"- {node['name']} "
                f"({node['type']})"
            )

        print("\nRelationships:")

        for edge in self.edges:
            print(
                f"- {edge['source']} "
                f"--[{edge['relationship']}]--> "
                f"{edge['target']}"
            )