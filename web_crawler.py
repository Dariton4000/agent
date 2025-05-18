import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

async def crawl(url):
    """
    Asynchronously crawls a web page and returns its content in markdown format.
    
    Args:
        url: The URL of the web page to crawl.
    
    Returns:
        The crawled page content as a markdown-formatted string.
    """
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
    Synchronously crawls a web page and returns its content in markdown format.
    
    Args:
        url: The URL of the web page to crawl.
    
    Returns:
        The crawled page content as a markdown string.
    """
    return asyncio.run(crawl(url))
