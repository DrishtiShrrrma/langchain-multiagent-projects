#=================
# Import Libraries
#=================

import streamlit as st
from crewai import Agent, Task, Crew
import os
from langchain_cohere import ChatCohere
from fpdf import FPDF
import pandas as pd
import plotly.express as px
import PyPDF2
from textblob import TextBlob
import time

#=================
# Add Streamlit Components
#=================

# Background
page_bg_img = '''
<style>
.stApp  {
background-image: url("https://images.all-free-download.com/images/graphiclarge/abstract_bright_corporate_background_310453.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title and Sidebar
st.title("AI Business Consultant")

image_url = "https://cdn-icons-png.flaticon.com/512/1998/1998614.png"
st.sidebar.image(image_url, caption="", use_column_width=True)
st.sidebar.write("AI Multi-Agent system for Business Insights, Analysis, and Recommendations.")

# Optional Customization Toggle
customization_enabled = st.sidebar.checkbox("Enable Agent Goal Customization", value=False)

# Default or Customizable Agent Goals
if customization_enabled:
    st.sidebar.markdown("### Customize Agent Goals")
    planner_goal = st.sidebar.text_area("Planner Goal", value="Plan engaging and factually accurate content about the topic.")
    writer_goal = st.sidebar.text_area("Writer Goal", value="Write insightful and engaging content based on the topic.")
    analyst_goal = st.sidebar.text_area("Analyst Goal", value="Perform statistical analysis to extract actionable insights.")
else:
    planner_goal = "Plan engaging and factually accurate content about the topic."
    writer_goal = "Write insightful and engaging content based on the topic."
    analyst_goal = "Perform statistical analysis to extract actionable insights."

# Multi-Topic Input
topics = st.text_area('Enter Business Search Areas (comma-separated)', value="Artificial Intelligence, Blockchain")
stakeholder = st.text_input('Enter The Stakeholder Team', value="Executives")

# Optional PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file for analysis (optional)", type=["pdf"])

# Extract Text from PDF
pdf_content = ""
if uploaded_file:
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            pdf_content += page.extract_text()
        st.success("PDF content extracted successfully!")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")

#=================
# LLM Object and API Key
#=================
os.environ["COHERE_API_KEY"] = "your_actual_cohere_api_key"
llm = ChatCohere()

#=================
# Crew Agents
#=================

planner = Agent(
    role="Business Consultant",
    goal=planner_goal,
    backstory="You're tasked with providing business insights and suggestions for the stakeholder team: {stakeholder}.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

writer = Agent(
    role="Business Writer",
    goal=writer_goal,
    backstory="You draft professional insights based on inputs from the Business Consultant and Data Analyst, tailored for: {stakeholder}.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

analyst = Agent(
    role="Data Analyst",
    goal=analyst_goal,
    backstory="Provide comprehensive analysis based on current trends and inputs from the Business Consultant.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

#=================
# Crew Tasks
#=================

plan = Task(
    description="Research and provide actionable business insights.",
    expected_output="A comprehensive consultancy outline with insights, analysis, and suggestions.",
    agent=planner
)

write = Task(
    description="Draft a professional business insights document.",
    expected_output="A professional document with actionable insights tailored for {stakeholder}.",
    agent=writer
)

analyse = Task(
    description="Perform statistical analysis and present actionable findings.",
    expected_output="A clear statistical analysis report for {stakeholder}.",
    agent=analyst
)

#=================
# Execution
#=================

crew = Crew(
    agents=[planner, analyst, writer],
    tasks=[plan, analyse, write],
    verbose=2
)

def generate_report(result):
    """Generate a PDF report from the Crew output."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", size=20, style="B")
    pdf.cell(200, 10, txt="AI Business Consultant Report", ln=True, align="C")

    # Add result content
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=result)

    # Save PDF
    report_path = "Business_Insights_Report.pdf"
    pdf.output(report_path)
    return report_path


def analyze_sentiment(text):
    """Perform sentiment analysis using TextBlob."""
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return f"Sentiment: {'Positive' if sentiment.polarity > 0 else 'Negative' if sentiment.polarity < 0 else 'Neutral'}, Polarity: {sentiment.polarity:.2f}"

if st.button("Run Analysis"):
    with st.spinner('Gathering insights...'):
        try:
            # Start timing
            start_time = time.time()

            # Combine user inputs and PDF content
            combined_content = f"Topics: {topics}\nStakeholder: {stakeholder}\n\nPDF Content:\n{pdf_content}" if pdf_content else f"Topics: {topics}\nStakeholder: {stakeholder}"

            # Run Crew Workflow
            result = crew.kickoff(inputs={"topic": combined_content, "stakeholder": stakeholder})
            
            # Task Execution Time
            end_time = time.time()
            st.success(f"Analysis completed in {end_time - start_time:.2f} seconds!")

            # Display Results
            st.markdown("### Insights and Analysis")
            st.write(result)

            # Sentiment Analysis
            sentiment_result = analyze_sentiment(result)
            st.markdown(f"### Sentiment Analysis\n{sentiment_result}")

            # Visualization Example
            st.markdown("### Data Visualization Example")
            data = pd.DataFrame({
                "Category": ["Trend 1", "Trend 2", "Trend 3"],
                "Value": [25, 50, 75]
            })
            fig = px.bar(data, x="Category", y="Value", title="Sample Market Trends")
            st.plotly_chart(fig)

            # Report Generation
            report_path = generate_report(result)
            with open(report_path, "rb") as file:
                st.download_button(
                    label="Download Report as PDF",
                    data=file,
                    file_name="Business_Insights_Report.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
