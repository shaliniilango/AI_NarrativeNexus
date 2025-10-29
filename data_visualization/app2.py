import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import re
from collections import Counter

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(layout='wide', page_title='Text Insight & Visualization Dashboard')

# -------------------- LOAD NLP MODEL -------------------- #
nlp = spacy.load("en_core_web_sm")

st.title("🧠 Universal Text Insight & Visualization Dashboard")
st.markdown("Paste **any text (article, report, paragraph, etc.)** below to automatically extract and visualize insights.")

# -------------------- TEXT INPUT -------------------- #
text_input = st.text_area("📄 Paste your text here:", height=300)

# -------------------- HELPER: CLEAN AMOUNT -------------------- #
def clean_amount(x):
    if pd.isna(x):
        return 0
    x = str(x).replace('₹', '').replace(',', '').replace('crore', '').replace('Cr', '').strip()
    try:
        return float(x)
    except:
        if 'L' in x or 'lakh' in x.lower():
            digits = ''.join([c for c in x if c.isdigit() or c == '.'])
            return float(digits)/100
        return 0

# -------------------- DATA EXTRACTION -------------------- #
def extract_data_from_text(text):
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append((ent.text.strip(), ent.label_))

    df = pd.DataFrame(entities, columns=["Entity", "Label"])

    # Try extracting money amounts for numeric visualization
    df_money = df[df['Label'] == 'MONEY'].copy()
    if not df_money.empty:
        df_money['Cleaned_Amount'] = df_money['Entity'].apply(clean_amount)

    return df, df_money

# -------------------- PROCESSING -------------------- #
if st.button("Analyze Text"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        df, df_money = extract_data_from_text(text_input)

        if df.empty:
            st.error("No entities detected in the text.")
        else:
            st.subheader("📋 Extracted Entities")
            st.dataframe(df)

            # -------------------- ENTITY DISTRIBUTION -------------------- #
            st.markdown("### 📊 Entity Type Distribution")
            label_counts = df['Label'].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(label_counts)
            with col2:
                fig1, ax1 = plt.subplots()
                ax1.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
                st.pyplot(fig1)

            # -------------------- MOST COMMON ENTITIES -------------------- #
            st.markdown("### 🏆 Top 10 Most Common Entities")
            top_entities = Counter(df['Entity']).most_common(10)
            if top_entities:
                top_df = pd.DataFrame(top_entities, columns=['Entity', 'Frequency']).set_index('Entity')
                st.bar_chart(top_df)
            else:
                st.info("No frequent entities found.")

            # -------------------- MONEY INSIGHTS -------------------- #
            if not df_money.empty and df_money['Cleaned_Amount'].sum() > 0:
                st.markdown("### 💰 Monetary Mentions Analysis")

                total_funds = df_money['Cleaned_Amount'].sum()
                avg_funds = df_money['Cleaned_Amount'].mean()
                max_fund = df_money['Cleaned_Amount'].max()

                col3, col4, col5 = st.columns(3)
                col3.metric("Total Monetary Mentions (approx)", f"{total_funds:,.2f} Cr")
                col4.metric("Average Mentioned Amount", f"{avg_funds:,.2f} Cr")
                col5.metric("Max Mentioned Amount", f"{max_fund:,.2f} Cr")

                fig2, ax2 = plt.subplots()
                ax2.plot(range(len(df_money)), df_money['Cleaned_Amount'], marker='o')
                ax2.set_xlabel("Mention Index")
                ax2.set_ylabel("Amount (Cr)")
                ax2.set_title("Sequence of Monetary Mentions")
                st.pyplot(fig2)
            else:
                st.info("No monetary values detected or couldn't parse numeric values.")

            # -------------------- DATE / YEAR TREND -------------------- #
            date_entities = df[df['Label'] == 'DATE']
            if not date_entities.empty:
                st.markdown("### 📅 Date Mentions Trend")
                years = []
                for d in date_entities['Entity']:
                    match = re.search(r"(20\d{2}|19\d{2})", d)
                    if match:
                        years.append(int(match.group(1)))
                if years:
                    year_series = pd.Series(years).value_counts().sort_index()
                    fig3, ax3 = plt.subplots()
                    ax3.plot(year_series.index, year_series.values, marker='o')
                    ax3.set_xlabel("Year")
                    ax3.set_ylabel("Frequency")
                    st.pyplot(fig3)
                else:
                    st.info("No valid year mentions found.")
            else:
                st.info("No date-related information found.")

            # -------------------- LOCATION INSIGHTS -------------------- #
            gpe_entities = df[df['Label'] == 'GPE']['Entity']
            if not gpe_entities.empty:
                st.markdown("### 🌍 Top Mentioned Locations")
                top_cities = gpe_entities.value_counts().head(5)
                fig4, ax4 = plt.subplots()
                ax4.pie(top_cities.values, labels=top_cities.index, autopct='%1.1f%%', startangle=90)
                st.pyplot(fig4)
            else:
                st.info("No location mentions found.")

            # -------------------- ORGANIZATION INSIGHTS -------------------- #
            org_entities = df[df['Label'] == 'ORG']['Entity']
            if not org_entities.empty:
                st.markdown("### 🏢 Top Mentioned Organizations")
                org_counts = org_entities.value_counts().head(10)
                st.bar_chart(org_counts)
            else:
                st.info("No organization mentions found.")

            # -------------------- SUMMARY -------------------- #
            st.markdown("---")
            st.markdown("### 💡 Smart Summary")
            summary = []
            if not df_money.empty:
                summary.append(f"Detected {len(df_money)} monetary mentions.")
            if not org_entities.empty:
                summary.append(f"{len(org_entities.unique())} organizations mentioned.")
            if not gpe_entities.empty:
                summary.append(f"{len(gpe_entities.unique())} locations found.")
            if len(summary) == 0:
                st.write("No specific structured patterns detected.")
            else:
                st.write(" ".join(summary))
