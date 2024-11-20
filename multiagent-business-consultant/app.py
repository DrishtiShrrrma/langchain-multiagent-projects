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
st.sidebar.write("This AI Business Consultant uses an AI Multi-Agent system to deliver insights, statistical analysis, and up-to-date information about business topics.")

# Text Inputs
business = st.text_input('Enter The Required Business Search Area', value="Artificial Intelligence")
stakeholder = st.text_input('Enter The Stakeholder Team', value="Executives")

# PDF Upload
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
    goal="Plan engaging and factually accurate content about the topic: {topic}",
    backstory="You're tasked with providing business insights and suggestions for the stakeholder team: {stakeholder}.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

writer = Agent(
    role="Business Writer",
    goal="Write insightful and engaging content based on the topic: {topic}",
    backstory="You draft professional insights based on inputs from the Business Consultant and Data Analyst, tailored for: {stakeholder}.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

analyst = Agent(
    role="Data Analyst",
    goal="Perform statistical analysis to extract actionable insights for the topic: {topic}",
    backstory="Provide comprehensive analysis based on current trends and inputs from the Business Consultant.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

#=================
# Crew Tasks
#=================

plan = Task(
    description=(
        "1. Research the latest trends, key players, and noteworthy news for {topic}.\n"
        "2. Provide a structured outline and actionable insights.\n"
        "3. Include considerations for international markets."
    ),
    expected_output="A comprehensive consultancy outline with insights, analysis, and suggestions.",
    agent=planner
)

write = Task(
    description=(
        "1. Use the consultant's plan to write a professional document.\n"
        "2. Ensure the document is polished, concise, and engaging for {stakeholder}.\n"
        "3. Limit the length to 200 words with supporting visuals."
    ),
    expected_output="A professional document with actionable insights tailored for {stakeholder}.",
    agent=writer
)

analyse = Task(
    description=(
        "1. Perform statistical analysis using the consultant's plan.\n"
        "2. Present the findings in a format useful for the Business Writer.\n"
        "3. Collaborate to ensure alignment on the final output."
    ),
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


if st.button("Run Analysis"):
    with st.spinner('Gathering insights...'):
        try:
            # Combine user inputs and PDF content
            combined_content = f"Topic: {business}\nStakeholder: {stakeholder}\n\nPDF Content:\n{pdf_content}"
            
            # Run Crew Workflow
            result = crew.kickoff(inputs={"topic": combined_content, "stakeholder": stakeholder})
            
            # Display Results
            st.markdown("### Insights and Analysis")
            st.write(result)

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

