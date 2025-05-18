import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

async def crawl(url):
    browser_config = BrowserConfig()  # Default browser configuration
    run_config = CrawlerRunConfig()   # Default crawl run configuration
    print("Crawling URL:", url)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_config
        )
        return(result.markdown)  # Print clean markdown content

def crawl_sync(url: str):
    """
    Run the async crawl function in a synchronous context and return markdown.
    """
    return asyncio.run(crawl(url))
