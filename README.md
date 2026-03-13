# NeoStats — Chat Agent

A conversational AI assistant that answers questions using your own documents and, when needed, the web. Ask in plain language and get answers with clear sources.

---

## What It Does

**NeoStats** is a **chat agent** that:

- **Talks about your documents** — Upload PDFs, text files, Word docs, or spreadsheets. Ask questions and get answers grounded in your files, with references to the exact parts used.
- **Searches the web when useful** — For things not in your documents (e.g. market news, recent updates), the agent can search the internet and summarize what it finds.
- **Remembers the conversation** — It keeps context within your chat so you can ask follow-up questions naturally.
- **Lets you choose answer style** — You can switch between **Concise** (short, to-the-point) and **Detailed** (more explanation and context) answers.

You can use it for financial documents, reports, or any content you upload — no need to be technical.

---

## How to Use

1. **Open the app** (locally or via the link if it’s deployed).
2. **Optional: upload a document** in the sidebar (PDF, TXT, DOCX, or CSV). Wait until you see a success message.
3. **Pick a response mode** in the sidebar: **Concise** or **Detailed**.
4. **Type your question** in the chat box and press Enter.
5. Read the answer; if the agent used documents or web results, open the **Sources** section under the reply to see where the information came from.

You can ask about your uploaded files only, or mix in general questions; the agent will use documents when relevant and search the web when needed.

---

## Run Locally

- Install dependencies: `pip install -r requirements.txt`
- Set your API key (e.g. in environment): `GROQ_API_KEY=your_key`
- Start the app: `streamlit run main.py`
- Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and create a new app from this repository.
3. Set the main file to **main.py**.
4. In the app’s **Settings → Secrets**, add your API key (e.g. `GROQ_API_KEY`).
5. Deploy; the app will redeploy automatically when you push new changes to the connected branch.

---


*NeoStats — Chat with your documents and the web.*
