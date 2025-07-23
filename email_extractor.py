import os
import re
import time
import random
import sqlite3
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
import threading

import pandas as pd
import requests
import httpx
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector

from dotenv import load_dotenv

load_dotenv()

EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
]
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
MAX_CONCURRENT_REQUESTS = 3
DELAY_RANGE = (1, 5)
JS_RENDER_TIMEOUT = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("email_extractor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmailExtractor:
    def __init__(self):
        self.ua = UserAgent()
        self.visited_urls = set()
        self.proxies = self._load_proxies()
        self.db_conn = self._init_db()
        self.driver = None
        self.session = None
        self.lock = threading.Lock()

    def _load_proxies(self) -> List[str]:
        proxies = []
        proxy_env = os.getenv("PROXY_LIST")
        if proxy_env:
            proxies.extend(proxy_env.split(","))
        try:
            with open("proxies.txt", "r") as f:
                proxies.extend([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            pass
        return [p if p.startswith('http') else f'http://{p}' for p in proxies]

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect("email_extractor.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraped_urls (
            url TEXT PRIMARY KEY,
            emails TEXT,
            timestamp DATETIME,
            status TEXT
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS extracted_emails (
            email TEXT PRIMARY KEY,
            source_url TEXT,
            timestamp DATETIME
        )
        """)
        conn.commit()
        return conn

    def _get_random_user_agent(self) -> str:
        try:
            return self.ua.random
        except:
            return random.choice(DEFAULT_USER_AGENTS)

    def _get_random_proxy(self) -> Optional[str]:
        if not self.proxies:
            return None
        return random.choice(self.proxies)

    def _get_selenium_driver(self):
        if self.driver:
            return self.driver
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user-agent={self._get_random_user_agent()}")
        proxy = self._get_random_proxy()
        if proxy:
            options.add_argument(f"--proxy-server={proxy}")
        self.driver = webdriver.Chrome(options=options)
        return self.driver

    def _close_selenium_driver(self):
        if self.driver:
            self.driver.quit()
            self.driver = None

    async def _get_async_session(self) -> ClientSession:
        if self.session and not self.session.closed:
            return self.session
        timeout = ClientTimeout(total=REQUEST_TIMEOUT)
        connector = TCPConnector(limit=MAX_CONCURRENT_REQUESTS, force_close=True)
        headers = {
            "User-Agent": self._get_random_user_agent()
        }
        self.session = ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers
        )
        return self.session

    async def _close_async_session(self):
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def _validate_email(self, email: str, domain_filter: Optional[str] = None) -> bool:
        if not re.fullmatch(EMAIL_REGEX, email):
            return False
        if any(term in email.lower() for term in ["noreply", "no-reply", "info", "support", "admin", "contact"]):
            return False
        if domain_filter and not email.endswith(domain_filter):
            return False
        return True

    def _extract_emails_from_text(self, text: str, domain_filter: Optional[str] = None) -> List[str]:
        emails = re.findall(EMAIL_REGEX, text)
        return list(set([email for email in emails if self._validate_email(email, domain_filter)]))

    async def _fetch_url_async(self, url: str, session: ClientSession) -> Optional[str]:
        try:
            proxy = self._get_random_proxy()
            async with session.get(url, proxy=proxy) as response:
                if response.status == 200:
                    return await response.text()
                logger.warning(f"Failed to fetch {url}: Status {response.status}")
        except Exception as e:
            logger.warning(f"Error fetching {url}: {str(e)}")
        return None

    def _fetch_url_with_selenium(self, url: str) -> Optional[str]:
        driver = self._get_selenium_driver()
        try:
            driver.get(url)
            WebDriverWait(driver, JS_RENDER_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            return driver.page_source
        except TimeoutException:
            logger.warning(f"Timeout while rendering {url}")
        except WebDriverException as e:
            logger.warning(f"Selenium error fetching {url}: {str(e)}")
        return None

    def _fetch_url(self, url: str) -> Optional[str]:
        for attempt in range(MAX_RETRIES):
            try:
                headers = {
                    "User-Agent": self._get_random_user_agent()
                }
                proxy = self._get_random_proxy()
                proxies = {"http": proxy, "https": proxy} if proxy else None
                response = requests.get(
                    url,
                    headers=headers,
                    proxies=proxies,
                    timeout=REQUEST_TIMEOUT
                )
                if response.status_code == 200:
                    return response.text
                logger.warning(f"Attempt {attempt + 1}: Failed to fetch {url}: Status {response.status_code}")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: Error fetching {url}: {str(e)}")
            time.sleep(random.uniform(*DELAY_RANGE))
        logger.info(f"Trying Selenium for {url}")
        return self._fetch_url_with_selenium(url)

    def _find_contact_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        contact_keywords = ["contact", "about", "team", "people", "staff", "careers"]
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].lower()
            text = a.get_text().lower()
            if any(keyword in href or keyword in text for keyword in contact_keywords):
                if href.startswith("/"):
                    full_url = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                    links.add(full_url)
                elif href.startswith("http"):
                    links.add(href)
        return list(links)

    async def _scrape_emails_from_page(self, url: str, domain_filter: Optional[str] = None, recursive: bool = False) -> List[str]:
        if url in self.visited_urls:
            return []
        self.visited_urls.add(url)
        logger.info(f"Scraping: {url}")
        with self.lock:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT emails FROM scraped_urls WHERE url = ?", (url,))
            result = cursor.fetchone()
            if result:
                logger.info(f"Found cached result for {url}")
                return result[0].split(",") if result[0] else []
        await asyncio.sleep(random.uniform(*DELAY_RANGE))
        session = await self._get_async_session()
        html = await self._fetch_url_async(url, session)
        if not html:
            with self.lock:
                cursor.execute(
                    "INSERT OR REPLACE INTO scraped_urls (url, emails, timestamp, status) VALUES (?, ?, ?, ?)",
                    (url, "", datetime.now(), "failed")
                )
                self.db_conn.commit()
            return []
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        emails = self._extract_emails_from_text(text, domain_filter)
        if recursive and len(emails) < 3:
            contact_links = self._find_contact_links(soup, url)
            logger.info(f"Found {len(contact_links)} contact links on {url}")
            tasks = []
            for link in contact_links:
                if link not in self.visited_urls:
                    tasks.append(self._scrape_emails_from_page(link, domain_filter, False))
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, list):
                        emails.extend(result)
        unique_emails = list(set(emails))
        with self.lock:
            cursor.execute(
                "INSERT OR REPLACE INTO scraped_urls (url, emails, timestamp, status) VALUES (?, ?, ?, ?)",
                (url, ",".join(unique_emails), datetime.now(), "success")
            )
            for email in unique_emails:
                cursor.execute(
                    "INSERT OR IGNORE INTO extracted_emails (email, source_url, timestamp) VALUES (?, ?, ?)",
                    (email, url, datetime.now())
                )
            self.db_conn.commit()
        return unique_emails

    def _scrape_emails_from_page_sync(self, url: str, domain_filter: Optional[str] = None, recursive: bool = False) -> List[str]:
        if url in self.visited_urls:
            return []
        self.visited_urls.add(url)
        logger.info(f"Scraping: {url}")
        with self.lock:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT emails FROM scraped_urls WHERE url = ?", (url,))
            result = cursor.fetchone()
            if result:
                logger.info(f"Found cached result for {url}")
                return result[0].split(",") if result[0] else []
        time.sleep(random.uniform(*DELAY_RANGE))
        html = self._fetch_url(url)
        if not html:
            with self.lock:
                cursor.execute(
                    "INSERT OR REPLACE INTO scraped_urls (url, emails, timestamp, status) VALUES (?, ?, ?, ?)",
                    (url, "", datetime.now(), "failed")
                )
                self.db_conn.commit()
            return []
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        emails = self._extract_emails_from_text(text, domain_filter)
        if recursive and len(emails) < 3:
            contact_links = self._find_contact_links(soup, url)
            logger.info(f"Found {len(contact_links)} contact links on {url}")
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                futures = []
                for link in contact_links:
                    if link not in self.visited_urls:
                        futures.append(executor.submit(
                            self._scrape_emails_from_page_sync,
                            link, domain_filter, False
                        ))
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        emails.extend(result)
                    except Exception as e:
                        logger.error(f"Error scraping contact page: {str(e)}")
        unique_emails = list(set(emails))
        with self.lock:
            cursor.execute(
                "INSERT OR REPLACE INTO scraped_urls (url, emails, timestamp, status) VALUES (?, ?, ?, ?)",
                (url, ",".join(unique_emails), datetime.now(), "success")
            )
            for email in unique_emails:
                cursor.execute(
                    "INSERT OR IGNORE INTO extracted_emails (email, source_url, timestamp) VALUES (?, ?, ?)",
                    (email, url, datetime.now())
                )
            self.db_conn.commit()
        return unique_emails

    async def scrape_urls_async(self, urls: List[str], domain_filter: Optional[str] = None, recursive: bool = False) -> Dict[str, List[str]]:
        results = {}
        try:
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            async def limited_scrape(url):
                async with semaphore:
                    emails = await self._scrape_emails_from_page(url, domain_filter, recursive)
                    return url, emails
            tasks = [limited_scrape(url) for url in urls if url not in self.visited_urls]
            for task in asyncio.as_completed(tasks):
                url, emails = await task
                results[url] = emails
                logger.info(f"Found {len(emails)} emails on {url}")
        finally:
            await self._close_async_session()
        return results

    def scrape_urls_sync(self, urls: List[str], domain_filter: Optional[str] = None, recursive: bool = False) -> Dict[str, List[str]]:
        results = {}
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            futures = {
                executor.submit(
                    self._scrape_emails_from_page_sync,
                    url, domain_filter, recursive
                ): url for url in urls if url not in self.visited_urls
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping URLs"):
                url = futures[future]
                try:
                    emails = future.result()
                    results[url] = emails
                    logger.info(f"Found {len(emails)} emails on {url}")
                except Exception as e:
                    logger.error(f"Error scraping {url}: {str(e)}")
                    results[url] = []
        return results

    def export_results(self, output_format: str = "csv") -> str:
        with self.lock:
            cursor = self.db_conn.cursor()
            cursor.execute("""
            SELECT email, source_url, timestamp 
            FROM extracted_emails 
            ORDER BY timestamp DESC
            """)
            emails_data = cursor.fetchall()
            cursor.execute("""
            SELECT url, status, timestamp, LENGTH(emails) as email_count 
            FROM scraped_urls 
            ORDER BY timestamp DESC
            """)
            urls_data = cursor.fetchall()
            emails_df = pd.DataFrame(emails_data, columns=["email", "source_url", "timestamp"])
            urls_df = pd.DataFrame(urls_data, columns=["url", "status", "timestamp", "email_count"])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_format == "csv":
                emails_df.to_csv("output.csv", index=False)
                return "Emails saved to output.csv"
            elif output_format == "excel":
                emails_df.to_excel(f"output_{timestamp}.xlsx", index=False)
                return f"Emails saved to output_{timestamp}.xlsx"
            elif output_format == "json":
                emails_df.to_json(f"output_{timestamp}.json", orient="records", indent=2)
                return f"Emails saved to output_{timestamp}.json"
            else:
                return "Invalid output format. Use 'csv', 'excel', or 'json'."

    def close(self):
        self._close_selenium_driver()
        if self.db_conn:
            self.db_conn.close()
        async def async_close():
            await self._close_async_session()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(async_close())
            else:
                loop.run_until_complete(async_close())
        except:
            pass

    def __del__(self):
        self.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Email Extractor Tool")
    parser.add_argument("input_file", help="CSV file containing URLs")
    parser.add_argument("--url_column", default="url", help="Column name containing URLs")
    parser.add_argument("--domain_filter", help="Filter emails by domain (e.g., 'company.com')")
    parser.add_argument("--recursive", action="store_true", help="Enable recursive scraping")
    parser.add_argument("--async_mode", action="store_true", help="Use async mode")
    parser.add_argument("--output", default="csv", choices=["csv", "excel", "json"], help="Output format")
    args = parser.parse_args()
    try:
        df = pd.read_csv(args.input_file)
        urls = df[args.url_column].dropna().unique().tolist()
        logger.info(f"Loaded {len(urls)} unique URLs from {args.input_file}")
    except Exception as e:
        logger.error(f"Error reading input file: {str(e)}")
        return
    extractor = EmailExtractor()
    try:
        start_time = time.time()
        if args.async_mode:
            async def async_main():
                return await extractor.scrape_urls_async(
                    urls, 
                    domain_filter=args.domain_filter,
                    recursive=args.recursive
                )
            results = asyncio.run(async_main())
        else:
            results = extractor.scrape_urls_sync(
                urls,
                domain_filter=args.domain_filter,
                recursive=args.recursive
            )
        export_result = extractor.export_results(args.output)
        logger.info(export_result)
        total_emails = sum(len(emails) for emails in results.values())
        logger.info(f"Extracted {total_emails} unique emails from {len(urls)} URLs in {time.time() - start_time:.2f} seconds")
    except KeyboardInterrupt:
        logger.info("Process interrupted. Saving progress...")
        extractor.export_results(args.output)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        extractor.close()

if __name__ == "__main__":
    main()