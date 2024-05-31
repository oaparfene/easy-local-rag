import torch
import ollama
import os
from openai import OpenAI
import argparse
import json

def remove_newlines(arr):
    return [s.replace("\n", "") for s in arr if s.replace("\n", "")]

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=5):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})
   
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})
    
    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        print(PINK + "Original Query: " + user_input + RESET_COLOR)
        print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query = user_input
    
    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"] = user_input_with_context
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content

# Parse command-line arguments
print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
args = parser.parse_args()

# Configuration for the Ollama API client
print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='llama3'
)

# Load the vault content
print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
#vault_content = ['MAJIIC Multi-sensor Aerospace-ground Joi nt ISR Interoperability Coalition Introduction The Multi-Sensor Aerospace-Ground Joint Intelligence, Surveillance and Reconnaissance (ISR) interoperability coalition (MAJIIC) project is a multinational effort to maximise the military utility of surveillance and reconnaissance resources through the development and evaluation of operational and technical means for interoperability of a wide range of ISR assets.In close cooperation with industry, the nations participating in MAJIIC are Canada, France, Germany, Italy, Netherlands, Norway, Spain, United Kingdom and the United States of America.Th e nations have appointed the NATO Consultation, Command and Control Agency (NC3A) as a facilitator for the project and to provide overall technical management.Organisation MAJIIC was established as a project under the multinational coalition surveillance and reconnaissance memorandum of understanding (CSR MOU).', 'Overall leadership is performed by a management team consisting of national representatives, while a group of national project officers (NPOs) handles day-to-day project execution. The project is further organised into an operational, an architectural a nd a technical working group, each of which reports to the NPOs.MAJIIC Management Team National Project Officers Architecture Working GroupOperational Working GroupTechnical Working GroupTechnical ManagerMAJIIC Management Team National Project Officers Architecture Working GroupOperational Working GroupTechnical Working GroupTechnical Manager MAJIIC Project Organisation Project Aims The primary aim of the MAJIIC project is to improve the commanders situation awareness through collaborative employment and use of interoperable ISR sensor and exploitation capabilities.Enhanced situation awareness enabled by MAJIIC capabilities.To achieve this, MAJIIC will address interoperability from three primary perspectives: 1.', 'Operational , including development and demonstration of concepts of employment (CONEMP) and tactics, techniques and procedures (TTP) for collaborative employment and use of coalition ISR assets in support of military missions. MAJIIC will also support incorporation of these operational documents into NATO and the nations 2.Architectural , including development of procedures and technology for sharing ISR data and information, system architecture design principles, tools and technology for collaboration, and tools for managing coalition ISR assets 3.Technical , including definition and development of key data formats and protocols for the various sensor and data types, tools to support common geo-registration, and data exploitation.', 'Approach to interoperability The MAJIIC project addresses the ability to collaboratively employ and exchange data from a wide variety of ISR sensors and sensor types in a network-enabled manner, including close coupling between the ISR assets and the NATO and national command and control (C2) environments. Operational Foundation To ensure that the project has the strongest possible operational foundation, the efforts under MAJIIC will be guided by operational doctrine in the form of CONEMP, TTP, and other requirements and guidelines.This doctrine will be developed by operational expertise from the participating nations working in close cooperation with NATO commands and liaising with a wide range of NATO, multinational and national activities and programmes.', 'Flexible and Wide-Reaching Approach MAJIIC will address interoperability in a flexible and wide- reaching manner, ranging from small tactical systems usually assigned to tactical commands and all the way up to highly capable strategic multi-user systems. Although the name of the project indicates an emphasis on aerospace- borne ISR systems, the project aims at addressing any sensor platform category, in cluding space-based, airborne, ground-based or maritime, as well as manned and unmanned subsets of these.The coalition ISR sensor environment Prepared by Lars Nesse, NC3A, October 2006 The sensor data types addressed in MAJIIC include ground moving target indicator (GMTI) radar, synthetic aperture radar (SAR), electro-optical (EO) and infra-red (IR) imaging and video sensors, electronic warfare support measures (ESM) sensors, an d artillery locating radar.', 'EO still/video GMTI SAR IR still/videoMAJIIC Tracks and Locations ESM EO still/video EO still/video GMTI GMTI SAR SAR IR still/video IR still/videoMAJIIC Tracks and Locations Tracks and Locations ESM ESM ISR data types addressed by the MAJIIC project. Interoperability Principle MAJIIC aims to enable interoperability between ISR and C2 systems through the use of common interfaces for data formats and exchange mechanisms, leaving the inner workings of each national system outside of the scope of the project and only requiring minor external interface modifications to each system.', 'Common Formats / DescriptionsType A GDT Type C GDT Ground Exploitation CapabilitiesSensor Type DSensor Type A Land TOCSATCOM GDTSensor Type CSensor Type B Type B GDTComms Relay Shared Data Servers Ops / Intel Systems NATO Provided Nation ProvidedNATO C2ISR Systems National C2ISR Systems MAJIIC interoperability architecture in principle Each system will provide data to a ground station or another component that is connected to a common network structure, enabling exchange of data and information outside the boundaries of each system. Interfaces and Mechanisms The common formats and exchange mechanisms employed in MAJIIC will be based on NATO standardisation agreements (STANAGs).', 'For da ta formats, this includes: - STANAG 4545: EO, IR and SAR still imagery - STANAG 4607: GMTI data - STANAG 4609: EO and IR motion imagery (video) - STANAG 5516: Track and track management messages MAJIIC will assess a wide range of network-enabled architecture approaches for enabling exchange of NRT and archived data and information, including techniques such as broadcast, publish-subscribe and request-only. As part of this, MAJIIC has implemented an interface based on STANAG 4559 (NATO Standard ISR Library Interface) for metadata-based access to and retrieval of archived data from any Coalition Shared Database (CSD) throughout the interconnected MAJIIC environment.The project will continuously be testing the implemented STANAGs during simulated and live exercises, and will work in close cooperation with the STANAG communities to ensure that problems and issues arising can be addressed in future updates to each STA NAG.', 'This effort will include development and validation of implementation guidelines to supplement those existing for each STANAG. In areas where no STANAG is available, such as Instant Messaging tools for distributed operator collaboration, the project will assess widely used commercial standards for potential use in coalition operations, such as the XMPP standard used in the Jabber chat tool.Networking and flexibility In order to be adaptable to real-world deployed operations, where the availability of terrestrial and satellite bandwidth might be scarce, MAJIIC will support interoperability using any network type or bandwidth, as well as any combination of networks and interconnections.This approach will include dissemination of near-r eal-time and archived data, the latter by using CSDs that are synchronised at the metadata level to provide full visibility into all archived data throughout the network independent of where the users are located.', 'MAJIIC deployed network interoperability Through this approach, MAJIIC will provide a true network-enabled capability enabling a wide variety of users at different locations and le vels of command to access and retrieve data in accordance with own tasks, needs, priorities, and prefer ences. The MAJIIC architecture is also compliant with the NATO Network-Enabled Capabilities (NNEC) initiative.Schedule The MAJIIC project started on 01 April 2005 and will last through March 2009.Throughout this period, the project will participate in or if necessary arrange at least one operationally-focussed exercise each year in order to test, verify, and refine the developed capabilities.This will include simulated as well as live exercises involving real ISR and C2 assets.', 'MAJIIC2005 2009 2008 2007 2006 Exercises SIMEX Live Fly TIE MAJEX MAJEX Contact: Management Team Chairman NPO Chairman Technical Manager Col Dave Neil Capt Lâl Marandin Mr Joe Ross / Mr Sean Midwood National Defence HQ, MGen George R Pearkes Building, Ottawa, Ontario K1A 0K2, Canada SPOTI/SMEA-OR, 18 rue du Dr Zamenhof, 92131 Issy-Les-Moulineaux Cedex, France NATO C3 Agency (NC3A), P.O. Box 174, 2501 CD The Hague, Netherlands Tel: +1 (613) 996 9747 Tel: +33 (0)1414 63329 Tel: +31 (0)70 374 3777 / 3694 Email: neil.dt@forces.gc.ca Email: lal.marandin@dga.defense.gouv.fr Email: joe.ross@nc3a.nato.int sean.midwood@nc3a.nato.int']
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

vault_content = remove_newlines(vault_content)
# Generate embeddings for the vault content using Ollama
print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
vault_embeddings = []
print("Looping through content in vault...\n\n", vault_content)
for content in vault_content:
    print("looping through content:\n\n ", content)
    #response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
    try:
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        print(response)
    except ollama._types.ResponseError as e:
        print(f"Exception occurred: {e}")
        print(f"Attributes: {dir(e)}")
        print(f"Error: {e.error}, Status Code: {e.status_code}")
        raise

    vault_embeddings.append(response["embedding"])

# Convert to tensor and print embeddings
print("Converting embeddings to tensor...")
vault_embeddings_tensor = torch.tensor(vault_embeddings) 
print("Embeddings for each line in the vault:")
print(vault_embeddings_tensor)

# Conversation loop
print("Starting conversation loop...")
conversation_history = []
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant infromation to the user query from outside the given context."

while True:
    user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break
    
    response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
    print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
