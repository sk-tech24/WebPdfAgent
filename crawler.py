import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from playwright.sync_api import sync_playwright
from langchain_core.documents import Document

class WebCrawler:
    def __init__(self, base_url, throttle_delay=2, use_dynamic=False):
        self.base_url = base_url
        self.visited_urls = set()
        self.to_visit_urls = set([base_url])
        self.throttle_delay = throttle_delay
        self.use_dynamic = use_dynamic
        self.documents = []

    def fetch_robots_txt(self):
        robots_url = urljoin(self.base_url, "/robots.txt")
        try:
            response = requests.get(robots_url, timeout=10)
            if response.status_code == 200:
                print(f"Found robots.txt:\n{response.text}")
                # Optional: Parse robots.txt to respect crawling rules
            else:
                print("No robots.txt found.")
        except Exception as e:
            print(f"Error fetching robots.txt: {e}")

    def fetch_sitemap(self):
        sitemap_url = urljoin(self.base_url, "/sitemap.xml")
        try:
            response = requests.get(sitemap_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'xml')
                urls = [loc.text for loc in soup.find_all("loc")]
                self.to_visit_urls.update(urls)
                print(f"Found {len(urls)} URLs in sitemap.")
            else:
                print("No sitemap.xml found.")
        except Exception as e:
            print(f"Error fetching sitemap: {e}")

    def extract_links(self, html, base_url):
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = urljoin(base_url, a_tag["href"])
            if href.startswith(self.base_url):  # Only include same-domain links
                links.add(href)
        return links

    def load_static_page(self, url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text
            else:
                raise Exception(f"Failed to load {url}, status code: {response.status_code}")
        except Exception as e:
            print(f"Error loading static page {url}: {e}")
            return None

    def load_dynamic_page(self, url):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=30000)
                content = page.content()
                browser.close()
                return content
        except Exception as e:
            print(f"Error loading dynamic page {url}: {e}")
            return None

    def load_webpage(self, url):
        if self.use_dynamic:
            return self.load_dynamic_page(url)
        else:
            return self.load_static_page(url)

    def extract_text_from_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        return soup.get_text(strip=True)

    def crawl(self):
        self.fetch_robots_txt()
        self.fetch_sitemap()

        while self.to_visit_urls:
            current_url = self.to_visit_urls.pop()
            if current_url in self.visited_urls:
                continue

            print(f"Crawling: {current_url}")
            html = self.load_webpage(current_url)
            if html:
                text = self.extract_text_from_html(html)
                self.documents.append({"url": current_url, "text": text})
                new_links = self.extract_links(html, current_url)
                self.to_visit_urls.update(new_links - self.visited_urls)

            self.visited_urls.add(current_url)

            # Throttling to avoid overwhelming the server
            import time
            time.sleep(self.throttle_delay)

        print(f"Crawling complete. Visited {len(self.visited_urls)} pages.")

    def get_documents(self):
        documents = []
        for doc in self.documents:
            documents.append(Document(
                    page_content=doc['text'],
                    metadata={"source": doc['url']}
                )
            )
        self.documents = documents
        return self.documents


# Initialize the crawler
crawler = WebCrawler(base_url="https://www.squadkin.com", throttle_delay=2, use_dynamic=True)

# Start crawling
crawler.crawl()

# Get the extracted documents
documents = crawler.get_documents()

# Print results
# for doc in documents:
    
#     print(f"URL: {doc['url']}")
#     print(f"Extracted Text: {doc['text'][:50]}...\n")
