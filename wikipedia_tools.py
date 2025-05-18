import requests

def wikipedia_search(query, limit=25):
    """
    Searches Wikipedia for a query and returns the title of the first result.
    
    Args:
    	query: The search string to query Wikipedia.
    	limit: Maximum number of search results to retrieve (default is 25).
    
    Returns:
    	The title of the first Wikipedia page found for the query, or None if no results are found.
    """
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
    'action': 'query',
    'format': 'json',
    'list': 'search',
    'srsearch': query,
    'srlimit': limit,
    }
    print("Searching Wikipedia for:", query)
    response = requests.get(url, params=params)
    data = response.json()
    results = data['query']['search']
    for r in results:
        return(r['title'])

def wikipedia(page):
    """
    Retrieves the plain text extract of a specified Wikipedia page.
    
    Args:
        page: The title of the Wikipedia page to fetch.
    
    Returns:
        The plain text extract of the page if available, otherwise a message indicating no extract was found.
    """
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'extracts',
        'explaintext': True,
        'titles': page
    }
    print("Fetching Wikipedia page:", page)
    response = requests.get(url, params=params)
    data = response.json()
    page_data = next(iter(data['query']['pages'].values()))
    if 'extract' in page_data:
        return(page_data['extract'])
    else:
        return("No extract found for the given page.")
