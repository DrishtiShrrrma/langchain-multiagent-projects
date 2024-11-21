#=================
# Import Libraries
#=================

import streamlit as st
from crewai import Agent, Task, Crew
import os
from langchain_groq import ChatGroq
from fpdf import FPDF
import pandas as pd
import plotly.express as px
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
st.sidebar.write(
    "This AI Business Consultant is built using an AI Multi-Agent system. "
    "It provides business insights, statistical analysis, and professional recommendations!"
)

# User Inputs
business = st.text_input('Enter The Required Business Search Area', value="Artificial Intelligence")
stakeholder = st.text_input('Enter The Stakeholder Team', value="Executives")

# Optional Customization
enable_customization = st.sidebar.checkbox("Enable Advanced Agent Customization")
if enable_customization:
    st.sidebar.markdown("### Customize Agent Goals")
    planner_goal = st.sidebar.text_area(
        "Planner Goal",
        value="Plan engaging and factually accurate content about the topic."
    )
    writer_goal = st.sidebar.text_area(
        "Writer Goal",
        value="Write insightful and engaging content based on the topic."
    )
    analyst_goal = st.sidebar.text_area(
        "Analyst Goal",
        value="Perform statistical analysis to extract actionable insights."
    )
else:
    planner_goal = "Plan engaging and factually accurate content about the topic."
    writer_goal = "Write insightful and engaging content based on the topic."
    analyst_goal = "Perform statistical analysis to extract actionable insights."

#=================
# LLM Object and API Key
#=================
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.2-90b-text-preview")

#=================
# Crew Agents
#=================

planner = Agent(
    role="Business Consultant",
    goal=planner_goal,
    backstory=(
        "You're tasked with providing insights about {topic} to the stakeholder: {stakeholder}. "
        "Your work will form the foundation for the Business Writer and Data Analyst."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

writer = Agent(
    role="Business Writer",
    goal=writer_goal,
    backstory=(
        "You will write a professional insights document about {topic}, "
        "based on the Business Consultant's plan and the Data Analyst's results."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

analyst = Agent(
    role="Data Analyst",
    goal=analyst_goal,
    backstory=(
        "You will perform statistical analysis on {topic}, based on the Business Consultant's plan. "
        "Your analysis will support the Business Writer's final document for {stakeholder}."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

#=================
# Crew Tasks
#=================

plan = Task(
    description=(
        "1. Research trends, key players, and noteworthy news for {topic}.\n"
        "2. Provide structured insights and actionable recommendations.\n"
        "3. Suggest strategies for dealing with international operators.\n"
        "4. Limit content to 500 words."
    ),
    expected_output="A comprehensive consultancy document with insights and recommendations.",
    agent=planner
)

write = Task(
    description=(
        "1. Use the Business Consultant's plan to write a professional document for {topic}.\n"
        "2. Structure the content with engaging sections and visuals.\n"
        "3. Ensure alignment with the stakeholder's goals.\n"
        "4. Limit the document to 200 words."
    ),
    expected_output="A professional document tailored for {stakeholder}.",
    agent=writer
)

analyse = Task(
    description=(
        "1. Perform statistical analysis to provide actionable insights for {topic}.\n"
        "2. Collaborate with the Business Consultant and Writer to align on key metrics.\n"
        "3. Present findings in a format suitable for inclusion in the final document."
    ),
    expected_output="A data-driven analysis tailored for {stakeholder}.",
    agent=analyst
)

#=================
# Execution
#=================

crew = Crew(
    agents=[planner, analyst, writer],
    tasks=[plan, analyse, write],
    verbose=True
)

def generate_pdf_report(result):
    """Generate a professional PDF report from the Crew output."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Arial", size=16, style="B")
    pdf.cell(200, 10, txt="AI Business Consultant Report", ln=True, align="C")
    pdf.ln(10)

    # Content
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=result)

    # Save PDF
    report_path = "Business_Insights_Report.pdf"
    pdf.output(report_path)
    return report_path

if st.button("Run Analysis"):
    with st.spinner('Executing analysis...'):
        try:
            start_time = time.time()
            result = crew.kickoff(inputs={"topic": business, "stakeholder": stakeholder})
            execution_time = time.time() - start_time

            # Display Results
            st.markdown("### Insights and Analysis")
            st.write(result)

            # Display Execution Time
            st.success(f"Analysis completed in {execution_time:.2f} seconds!")

            # Visualization Example
            st.markdown("### Data Visualization Example")
            data = pd.DataFrame({
                "Metric": ["Trend 1", "Trend 2", "Trend 3"],
                "Value": [45, 80, 65]
            })
            fig = px.bar(data, x="Metric", y="Value", title="Sample Metrics for Analysis")
            st.plotly_chart(fig)

            # Generate and Provide PDF Report
            report_path = generate_pdf_report(result)
            with open(report_path, "rb") as file:
                st.download_button(
                    label="Download Report as PDF",
                    data=file,
                    file_name="Business_Insights_Report.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
