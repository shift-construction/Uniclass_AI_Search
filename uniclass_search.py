import streamlit as st
import requests
import time
import random
import pandas as pd

# Constants
MAX_RETRIES = 3  # Maximum number of retries for the API call
BACKOFF_FACTOR = 2  # Factor to determine the backoff period

# Replace the following with your actual credentials and endpoint details
openai_api_key = st.secrets["openai_api_key"]
pinecone_api_key = st.secrets["pinecone_api_key"]
pinecone_index_url = st.secrets["pinecone_index_url"]

# Function to get the embedding from OpenAI
def get_openai_embedding(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "input": text,
        "model": "text-embedding-ada-002"
    }

    for attempt in range(MAX_RETRIES):
        response = requests.post('https://api.openai.com/v1/embeddings', headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        elif response.status_code == 429 or response.status_code >= 500:
            # If rate limited or server error, wait and retry
            time.sleep((BACKOFF_FACTOR ** attempt) + random.uniform(0, 1))
        else:
            # For other errors, raise exception without retrying
            raise Exception(f"OpenAI API error: {response.text}")
    raise Exception(f"OpenAI API request failed after {MAX_RETRIES} attempts")

# Function to send the embedding to Pinecone and get the results
def query_pinecone(embedding):
    headers = {
        "Api-Key": pinecone_api_key,
        "Content-Type": "application/json"
    }
    data = {
        "vector": embedding,
        "topK": 10,
        "includeValues": False,
        "includeMetadata": True,
        "namespace": "uniclass_codes"
    }
    url = f"https://{pinecone_index_url}/query"
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['matches']
    else:
        raise Exception(f"Pinecone API error: {response.text}")

# Set up the Streamlit app
def main():
    st.set_page_config(page_title="Uniclass Search Engine", page_icon=":mag:")

    # Header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://usercontent.one/wp/www.shift-construction.com/wp-content/uploads/2023/10/shift-grey-logo-white-text_small.png", width=100)  # Replace with the path to your logo image
    with col2:
        st.title("Uniclass Search Engine")

    # Search box
    with st.form(key='search_form'):
        search_query = st.text_input("Enter a search term", key="search_box")
        submit_button = st.form_submit_button(label='Search')
    
        # Additional text with hyperlink
    st.markdown("This is the prototype Uniclass search engine as outlined in [this article](https://medium.com/shift-construction/creating-an-ai-powered-uniclass-classification-engine-part-1-search-engine-a7ec50c756ab). Follow Shift Construction on [Medium](https://medium.com/shift-construction) or [LinkedIn](https://www.linkedin.com/company/98679370) to see how the search engine evolves.")

    # If search was triggered, get the embedding and then query Pinecone
    if submit_button and search_query:
        try:
            embedding = get_openai_embedding(search_query)
            results = query_pinecone(embedding)

            if results:
                # Display results in a table
                st.write("Search Results:")
                # Use a dataframe to display the results in a more tabular form
                result_data = [{"code": match["metadata"]["code"], "title": match["metadata"]["title"]} for match in results]
                df = pd.DataFrame(result_data)
                st.table(df)
            else:
                st.write("No results found for your query.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
