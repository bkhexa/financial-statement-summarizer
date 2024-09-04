import streamlit as st
import logging
import requests
from dotenv import load_dotenv
import os
import pandas as pd
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables from a .env file
load_dotenv()

# Retrieve the variables from environment
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_type = os.getenv("AZURE_OPENAI_API_TYPE")
FMP_API_KEY = os.getenv('FMP_API_KEY')
COMPANY_NAME = os.getenv('COMPANY_NAME')

# Configure logging to output informational and error messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title='AI Financial Analyst', layout="wide") 
def get_jsonparsed_data(url):
    """
    Fetches data from the provided URL and parses it as JSON.
    
    Args:
        url (str): The URL to fetch data from.

    Returns:
        dict: Parsed JSON data, or an empty dictionary if an error occurs.
    """
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raises an error for bad status codes
        data = r.json()
        logging.info(f"Data successfully retrieved from {url}")
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        return {}

def get_financial_statements(ticker, limit, period, statement_type):
    """
    Retrieves financial statements for the given ticker.

    Args:
        ticker (str): The stock ticker symbol.
        limit (int): The number of financial statements to retrieve.
        period (str): 'annual' or 'quarterly' period selection.
        statement_type (str): The type of financial statement ('Income Statement', 'Balance Sheet', or 'Cash Flow').

    Returns:
        pd.DataFrame: DataFrame containing the financial statements.
    """
    try:
        # Validate input to prevent misuse
        if not ticker.isalnum():
            st.error("Invalid ticker symbol. Please enter a valid alphanumeric ticker symbol.")
            return pd.DataFrame()

        # Construct the appropriate URL based on the statement type
        if statement_type == "Income Statement":
            url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={FMP_API_KEY}"
        elif statement_type == "Balance Sheet":
            url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={FMP_API_KEY}"
        elif statement_type == "Cash Flow":
            url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={FMP_API_KEY}"
        else:
            st.error("Invalid statement type selected.")
            return pd.DataFrame()

        # Fetch and parse the data from the API
        data = get_jsonparsed_data(url)
        if not data:
            st.error("Failed to retrieve data. Please check your API key and ticker.")
            return pd.DataFrame()

        # Extract the relevant reports based on the period
        if period == "annual":
            reports = data.get("annualReports", [])
        elif period == "quarterly":
            reports = data.get("quarterlyReports", [])
        else:
            st.error("Invalid period selected. Please choose either 'annual' or 'quarterly'.")
            return pd.DataFrame()

        # Return the data as a DataFrame if available
        if reports:
            logging.info(f"{len(reports)} financial statements retrieved for {ticker}")
            return pd.DataFrame(reports[:limit])
        else:
            st.error("No financial statements found for the selected ticker.")
            return pd.DataFrame()

    except Exception as e:
        logging.error(f"An error occurred while processing financial statements: {e}")
        st.error("An unexpected error occurred. Please try again.")
        return pd.DataFrame()

def generate_financial_summary(financial_statements, statement_type):
    """
    Generate a summary of financial statements using Azure OpenAI.

    Args:
        financial_statements (pd.DataFrame): DataFrame containing financial statements.
        statement_type (str): The type of financial statement being summarized.

    Returns:
        str: The generated financial summary.
    """
    summaries = []
    for i in range(len(financial_statements)):
        period_ending = financial_statements['fiscalDateEnding'][i]
        if statement_type == "Income Statement":
            summary = f"For the period ending {period_ending}, the company reported the following key figures: ..."
        elif statement_type == "Balance Sheet":
            summary = f"For the period ending {period_ending}, the company reported the following key balances: ..."
        elif statement_type == "Cash Flow":
            summary = f"For the period ending {period_ending}, the company reported the following cash flow details: ..."
        summaries.append(summary)

    all_summaries = "\n\n".join(summaries)

    # Logging summary generation for traceability
    logging.info(f"Generated financial summary for {statement_type}")

    # Initialize Azure OpenAI with the specified configuration
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=0,
        model_name=model_name,
        openai_api_base=api_base,
        openai_api_version=api_version,
        openai_api_type=api_type,
        openai_api_key=openai_api_key
    )

    # Define a prompt template for summarization
    prompt_template = """
    You are an AI financial analyst. Analyze the following financial statements data and provide insights:
    {summaries}

    For each period, write out the key metrics in detail and then analyze how these metrics have changed over time.
    """

    # Use the prompt template to create a prompt for the LLMChain
    prompt = PromptTemplate(input_variables=["summaries"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the LLMChain to generate the financial summary
    response = chain.run({"summaries": all_summaries})
    return response

def financial_statements():
    """
    Streamlit app function for displaying financial statements and generating summaries.
    """
    st.title('Financial Statements')

    # Select the type of financial statement
    statement_type = st.selectbox("Select financial statement type:", ["Income Statement", "Balance Sheet", "Cash Flow"])

    # Create columns for input controls
    col1, col2 = st.columns(2)

    with col1:
        # Select the period (Annual or Quarterly)
        period = st.selectbox("Select period:", ["Annual", "Quarterly"]).lower()

    with col2:
        # Input for the number of past financial statements to analyze
        limit = st.number_input("Number of past financial statements to analyze:", min_value=1, max_value=5, value=3)

    # Input for the company ticker symbol
    ticker = st.text_input("Please enter the company ticker:")

    # Button to trigger the analysis
    if st.button('Run'):
        if ticker is not None:
            ticker = ticker.upper()
            # Retrieve the financial statements based on the provided inputs
            financial_statements = get_financial_statements(ticker, limit, period, statement_type)

            # Display the retrieved financial statements
            with st.expander("View Financial Statements"):
                st.dataframe(financial_statements)

            # Generate and display the financial summary if data is available
            if not financial_statements.empty:
                financial_summary = generate_financial_summary(financial_statements, statement_type)

                st.write(f'Summary for {ticker}:')
                summary_html = """
                    <div style='font-family:  "Times New Roman", Times, serif; font-size: 16px; color: darkblue;'>
                        {}
                    </div>
                """.format(financial_summary.replace('\n', '<br>'))
                st.markdown(summary_html, unsafe_allow_html=True)
        else:
            st.write("Hey, it seems you are forgetting to type the Company Symbol here...")

@st.cache_data
def load_data(file_path):
    """
    Load data from a CSV file and prepare it for use.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    df = pd.read_csv(file_path, header=None)
    df.columns = df.iloc[0].str.strip()
    df = df[1:].reset_index(drop=True)
    return df

# Define the search function
def search_company_symbol(df, search_term, top_n=4):
    """
    Search for a company name or ticker symbol in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing company data.
        search_term (str): The search term (company name or ticker symbol).
        top_n (int): The maximum number of results to return.

    Returns:
        pd.DataFrame: DataFrame containing the search results.
    """
    # Perform a case-insensitive search in both columns
    result = df[(df['Company'].str.contains(search_term, case=False, na=False)) |
                (df['Symbol'].str.contains(search_term, case=False, na=False))]
    # Return only the Company and Symbol columns
    return result[['Company', 'Symbol']].head(top_n)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.sidebar.title('AI Financial Analyst')
    
    # Sidebar selection for different features
    app_mode = st.sidebar.selectbox("Choose your AI assistant:", ["Financial Statements"])

    # New Company and Symbol Search feature in the sidebar
    st.sidebar.subheader("Search for Company to get the Symbol")

    comapny_name = COMPANY_NAME

    if comapny_name is not None:
        df = load_data(comapny_name)
        
        # Display the search box with dynamic search
        search_term = st.sidebar.text_input("Enter company name to get the ticker: eg. amazon - AMZN", "")
        
        if search_term:  # Perform search as you type
            results = search_company_symbol(df, search_term)
            
            # Display the results
            if not results.empty:
                st.sidebar.write(f"Found {len(results)} Companies:")
                st.sidebar.write(f"Please copy the symbol and paste it in the company ticker...")
                st.sidebar.dataframe(results.to_dict(orient='records'))

                st.sidebar.info('Some Companies might not be available...', icon="ℹ️")
            else:
               st.sidebar.write("No matching results found.")
    else:
        st.warn("please type something to begin with")
    
    # Run the financial statements function if selected
    if app_mode == 'Financial Statements':
        financial_statements()

# Run the Streamlit app when the script is executed
if __name__ == '__main__':
    main()
