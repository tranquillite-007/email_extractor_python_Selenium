# 📧 Smart Email Extractor (Advanced Web Scraping Framework)

## 🚀 About the Project

This project is a robust, scalable, and production-ready email extraction tool built to crawl websites, navigate internal contact pages, and extract verified, relevant email addresses.

It was developed as part of a backend engineering and automation learning journey. The tool is ideal for:

- Collecting recruiter or company emails from a list of websites.
- Building lead-generation datasets for job/internship search.
- Crawling contact pages of tech companies to gather hiring contacts.
- Researching and enriching startup databases with verified emails.

**Features:**

- Multi-threaded synchronous and asynchronous scraping
- Selenium-based JavaScript rendering
- Proxy rotation and user-agent spoofing
- SQLite checkpointing and deduplication
- Recursive crawling of contact/about/team pages
- Clean, modular Python class

---

## 🔧 Tech Stack & Libraries

- **Python 3.11+**
- `aiohttp`, `httpx`, `requests` – HTTP fetching (async + sync)
- `Selenium` – Rendering JS-heavy websites
- `BeautifulSoup` – HTML parsing
- `SQLite3` – Local database for caching
- `ThreadPoolExecutor` – Parallelism
- `pandas` – Export to CSV/Excel/JSON
- `dotenv` – Proxy/environment configs

---

## 🧠 Skills Demonstrated

- Asynchronous programming with `asyncio`, `aiohttp`
- Web automation with Selenium
- Anti-blocking strategies: rotating proxies, user-agents, and delays
- Scalable, modular scraping architecture
- SQL-based data caching and deduplication
- CLI interface with `argparse`
- Logging and error handling
- Multi-format export (CSV, Excel, JSON)

---

## 💼 Use Cases

- **Job Search Automation:** Extract recruiter emails from tech websites or hiring pages.
- **Startup Research:** Gather contact information for potential partnerships.
- **Data Enrichment:** Supplement scraped company data with verified emails.
- **Lead Generation for B2B:** Generate warm leads.
- **Academic/Portfolio Projects:** Demonstrate backend, scraping, and data pipeline skills.

---

## 📁 Folder Structure

```
email_extractor/
│
├── email_extractor.py       # Main script
├── email_extractor.db       # SQLite database (auto-generated)
├── output.csv               # Final output of scraped emails
├── README.md                # This file
└── requirements.txt
```

---

## ⚡️ Quickstart & Running Commands

### 1. **Set up a virtual environment (Recommended)**

**Windows:**

```sh
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**

```sh
python3 -m venv venv
source venv/bin/activate
```

### 2. **Install requirements**

```sh
pip install -r requirements.txt
```

### 3. **Install Playwright and its dependencies (if using Playwright)**

```sh
playwright install
playwright install-deps
```

### 4. **Install WebDriver for Selenium (if using Selenium/Chrome)**

```sh
webdriver-manager install --drivers chrome
```

### 5. **Prepare your input CSV**

- Must have a column with URLs (default column name: `url`).

### 6. **Basic Usage**

```sh
python email_extractor.py input.csv
```

### 7. **Specify a different URL column**

```sh
python email_extractor.py input.csv --url_column website
```

### 8. **Filter emails by domain**

```sh
python email_extractor.py input.csv --domain_filter company.com
```

### 9. **Enable recursive scraping (follow contact/about/team links)**

```sh
python email_extractor.py input.csv --recursive
```

### 10. **Use asynchronous mode (faster for many URLs)**

```sh
python email_extractor.py input.csv --async_mode
```

### 11. **Export to Excel or JSON**

```sh
python email_extractor.py input.csv --output excel
python email_extractor.py input.csv --output json
```

---

## 📝 Example Input CSV

```csv
url
https://example.com
https://anothercompany.com
```

---

## 🛠️ Requirements

- Python 3.11+
- ChromeDriver (for Selenium, if scraping JS-heavy sites)
- See `requirements.txt` for all dependencies

---

## 🙋‍♂️ Questions?

Open an issue or reach out!
