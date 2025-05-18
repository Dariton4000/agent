import requests

def wikipedia_search(query, limit=25):
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
