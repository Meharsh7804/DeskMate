# DeskMate - Your Campus Professor Tracker

## 📌 Project Overview

DeskMate helps students locate professors on campus by checking their availability and seating information via an interactive chatbot. It eliminates the hassle of manually searching for faculty members by leveraging AI-powered search and response capabilities. Students can upload department timetables in PDF format and instantly ask questions about a professor’s schedule.<br> 

The system uses Google Gemini AI for intelligent responses and FAISS for quick and efficient vector searches. With a user-friendly Streamlit-based UI, DeskMate provides a seamless experience for students to find the right professor at the right time.

## 🚀 Features

- Upload department timetables (PDF)

- Ask queries about professor availability

- AI-powered responses using Google Gemini API

- Fast vector search with FAISS

- Streamlit-based interactive UI

## 🛠️ Installation Guide

1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
2️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/deskmate.git
cd deskmate
```
3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
4️⃣ Set Up API Key <br>
Create a .env file in the root directory and add:
```bash
GOOGLE_API_KEY=your_api_key_here
```
## 📂 Project Structure
```bash
📦 deskmate
├── 📜 main.py             # Streamlit app entry point
├── 📜 requirements.txt    # Required dependencies
├── 📜 .env                # API key configuration
└── 📂 faiss_index         # Local vector storage
```
▶️ Run DeskMate
```bash
streamlit run main.py
```
## ❓ Troubleshooting

### Invalid API Key? <br>
Check .env file.

### No Response from AI? <br>
Ensure internet access & valid API key.

### PDF Not Processing? <br>
Confirm file format and retry.

## 🛠️ Contributing

Pull requests are welcome! Follow the GitHub workflow:

- Fork the repository

- Create a new branch (git checkout -b feature-branch)

- Commit changes (git commit -m "Added new feature")

- Push (git push origin feature-branch)

- Open a pull request
