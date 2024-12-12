import os
from neo4j import GraphDatabase
import spacy
from langchain_community.document_loaders import WikipediaLoader
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import csv
import wikipediaapi


context_prompt = """
You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
Provide a set of Nodes in the form [ENTITY_ID, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES].
It is important that the ENTITY_ID_1 and ENTITY_ID_2 exists as nodes with a matching ENTITY_ID. If you can't pair a relationship with a pair of nodes don't add it.
When you find a node or relationship you want to add try to create a generic TYPE for it that describes the entity you can also think of it as a label. If multiple words are needed to form a TYPE, use '_' between them. I repeat, do not use spaces in the TYPE.
In your response, only give me the nodes and relationships in csv format. Both nodes and relationships should have a header row. Both nodes and relationships headers should start with the entity_id (node_id or relationship_id). They should be in the same csv file.
The property names should be a single word.
Do not include anything other text to the response.

Example:
Data: Alice lawyer and is 25 years old and Bob is her roommate since 2001. Bob works as a journalist. Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com.
Nodes: node_id,node_type,name,important_related_date,role ; alice,Person,Alice,12-12-1996,lawyer ; bob,Person,Bob,11-11-1987,journalist ; alicePage,Webpage,alice.com,7-7-2007,, ; bobPage,Webpage,bob.com,8-8-2008,, ;
Relationships: relationship_id,node_id_1,relationship_type,node_id_2,start_date ; 1,1,roommate,2,2001 ; 2,1,owns,3,7-7-2007, ; 3,2,owns,4,8-8-2008 ;

I want the the nodes in the exact following format :
node_id,node_type,name,birth_year,death_year,role,location

I want the relationships in the exact following format :
relationship_id,node_id_1,relationship_type,node_id_2,start_date

Here is the Data: 

"""

load_dotenv()
page_title = "Muhammad"
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")


def extract_wikipedia_content(page_title):
    # Extracts content from the specified Wikipedia page.
    # This function fetches the text content from a Wikipedia page using the `wikipediaapi` library.
    # It returns the page's text if it exists, otherwise it prints an error message.

    user_agent = "WikipediaAPI/0.5 (Academic Project; rayan.hanader@gmail.com)"

    wiki_wiki = wikipediaapi.Wikipedia(
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent=user_agent,
    )

    page = wiki_wiki.page(page_title)

    if page.exists():
        return page.text
    else:
        print("Page does not exist!")
        return None



def process_wikipedia_content(content):
    # Processes the Wikipedia content using the spaCy library.
    # This function loads the spaCy model "en_core_web_sm" (or downloads it if it's not available)
    # and filters out stop words from the content. It returns a filtered version of the content with stop words removed.
    
    if content:

        try:
            # Load the spaCy model
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If the model isn't found, provide instructions and attempt to download
            print("Model 'en_core_web_sm' not found. Downloading it now...")
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            
        doc = nlp(content)

        filtered_content = ""
        for sent in doc.sents:
            filtered_tokens = [token.text for token in sent if not token.is_stop]
            filtered_sentence = " ".join(filtered_tokens)
            filtered_content += filtered_sentence + "\n"
        return filtered_content



def chat_with_llm(prompt):
    # Interacts with the Generative AI model to generate content based on a prompt.
    # This function sends the provided prompt to Google's generative AI API (configured with the API key).
    # It then returns the generated text response.

    genai.configure(api_key=GOOGLE_GENAI_API_KEY)

    model = genai.GenerativeModel("learnlm-1.5-pro-experimental")
    response = model.generate_content(prompt)
    
    return response.text



def txt_file_from_str(string, filename="output.txt"):
    # Writes a given string to a text file.
    # This function saves the provided string into a text file with the specified filename.
    # The default filename is "output.txt".

    with open(filename, "w", encoding="UTF-8") as text_file:
        text_file.write(string)



def extract_entities_and_relationships(content):
    # Extracts entities and relationships from the given content by passing it through the LLM.
    # This function sends the content to a pre-defined prompt that interacts with the AI model to extract nodes and relationships.
    # It then returns the AI-generated CSV response.

    prompt = context_prompt + content
    response = chat_with_llm(prompt)
    return response



def Json_to_cypher(Json_response):
    # Converts the JSON response from the LLM into a Cypher query for Neo4j.
    # This function parses the JSON response containing nodes and relationships, and constructs Cypher queries
    # to create the nodes and relationships in Neo4j format.

    data = json.loads(Json_response)
    nodes = data["nodes"]
    relationships = data["relationships"]

    cypher_query = ""

    # For nodes
    for node in nodes:
        node_ID = node[0]
        node_type = node[1]
        properties = node[2]

        property_str = ", ".join(
            [f'{key}: "{value}"' for key, value in properties.items()]
        )

        cypher_query += f"CREATE ({node_ID}:{node_type} {{{property_str}}})\n"

    # For relationships
    for relationship in relationships:
        properties = relationship[3]
        property_str = ", ".join(
            [f'{key}: "{value}"' for key, value in properties.items()]
        )
        cypher_query += f"CREATE ({relationship[0]})-[:{relationship[1]} {{{property_str}}}]->({relationship[2]})\n"

    return cypher_query



def clean_csv(csv_response_txt):
    # Cleans and separates the CSV response of the LLM into nodes and relationships.
    # This function splits the CSV response by newlines and processes it to remove empty lines or unwanted characters.
    # It then writes the nodes and relationships to separate CSV files.

    csv_response = csv_response_txt.split("\n")
    print(csv_response)
    csv_response = [line for line in csv_response if line]
    print(csv_response)

    print("csv_response en parcours")
    for index, line in enumerate(csv_response):
        if "`" in line or line == "":
            csv_response.pop(index)
        if "relationship_id" in line:
            print(line)
            relationship_start = index
    print("fin parcours")

    nodes = csv_response[0 : relationship_start]
    relationships = csv_response[relationship_start :]
    
    # Write nodes to nodes.csv
    with open("./data/nodes.csv", "w", newline='', encoding='utf-8') as nodes_file:
        writer = csv.writer(nodes_file)
        for node in nodes:
            writer.writerow(node.split(","))

    # Write relationships to relationships.csv
    with open("./data/relationships.csv", "w", newline='', encoding='utf-8') as relationships_file:
        writer = csv.writer(relationships_file)
        for relationship in relationships:
            writer.writerow(relationship.split(","))

    return



def upload_csv_to_neo4j(uri, user, password, nodes_file, relationships_file):
    # Uploads nodes and relationships from CSV files to a Neo4j AuraDB instance.
    # This function connects to the Neo4j database using the provided credentials and CSV files,
    # and uploads the data using Cypher queries for creating nodes and relationships.


    driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_nodes(tx, row):
        node_type = row['node_type'].upper()
        query = (
            f"MERGE (n:{node_type} {{node_id: $node_id}})\n"
            "SET n.name = $name,\n"
            "    n.birth_year = CASE $birth_year WHEN '' THEN NULL ELSE toInteger($birth_year) END,\n"
            "    n.death_year = CASE $death_year WHEN '' THEN NULL ELSE toInteger($death_year) END,\n"
            "    n.role = CASE $role WHEN '' THEN NULL ELSE $role END,\n"
            "    n.location = CASE $location WHEN '' THEN NULL ELSE $location END"
        )
        tx.run(
            query,
            node_id=row["node_id"],
            name=row["name"],
            birth_year=row["birth_year"],
            death_year=row["death_year"],
            role=row["role"],
            location=row["location"],
        )

    def create_relationships(tx, row):
        query = (
            f"MATCH (a {{node_id: $node_id_1}}), (b {{node_id: $node_id_2}})\n"
            f"MERGE (a)-[r:{row['relationship_type'].upper()}]->(b)\n"
            "SET r.start_date = CASE $start_date WHEN '' THEN NULL ELSE toInteger($start_date) END"
        )
        tx.run(
            query,
            node_id_1=row["node_id_1"],
            node_id_2=row["node_id_2"],
            start_date=row["start_date"],
        )

    with driver.session() as session:
        # Load nodes
        with open(nodes_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                session.execute_write(create_nodes, row)

        # Load relationships
        with open(relationships_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                session.execute_write(create_relationships, row)

    driver.close()

def generate_graph(page_title):
    content = extract_wikipedia_content(page_title)
    csv_response = extract_entities_and_relationships(content)
    txt_file_from_str(csv_response, "./data/llm_csv_response.txt")
    clean_csv(csv_response)
    upload_csv_to_neo4j(uri, username, password, "./data/nodes.csv", "./data/relationships.csv")


def generate_graph_with_processed_content(page_title):
    content = extract_wikipedia_content(page_title)
    processed_content = process_wikipedia_content(content)
    csv_response = extract_entities_and_relationships(processed_content)
    txt_file_from_str(csv_response, "./data/llm_csv_response.txt")
    clean_csv(csv_response)
    upload_csv_to_neo4j(uri, username, password, "./data/nodes.csv", "./data/relationships.csv")

def run_cypher_query(query, driver):
    with driver.session() as session:
        session.run(query)

def generate_graph_from_manually_extracted_response(json_file):
    with open(json_file, "r") as file:
        Json_response = file.read()
    cypher_query = Json_to_cypher(Json_response)
    driver = GraphDatabase.driver(uri, auth=(username, password))
    run_cypher_query(cypher_query, driver)
    driver.close()

if __name__ == "__main__":


    generate_graph_with_processed_content("Muhammad")
    # generate_graph("Muhammad")
    # generate_graph_from_manually_extracted_response("data/google_learnlm_1.5_response.json")
    
    print("Done")