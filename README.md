# Knowledge Graph Extraction from Wikipedia

This project extracts entities and relationships from a Wikipedia page (e.g., Prophet of Islam "Muhammad") and stores them in a Neo4j graph database. The content is processed using natural language processing (NLP) techniques and Google's LearnLM 1.5 Pro Experimental model to generate nodes and relationships.

## Features
- Extracts entities and relationships from Wikipedia content
- Processes text using spaCy and OpenAI GPT-3
- Stores data in a Neo4j graph database
- Supports content splitting for large pages to handle token limits
- Handles data hallucinations using prompt engineering

## Technologies
- **Python**: Main programming language
- **spaCy**: Text preprocessing
- **Google's LearnLM 1.5 Pro**: For entity and relationship extraction
- **Neo4j**: Graph database
- **wikipedia-api**: To fetch Wikipedia content
