import wikipediaapi
import requests # library to handle requests
from bs4 import BeautifulSoup # library to parse HTML documents
import json

wiki_wiki = wikipediaapi.Wikipedia('de')

# extract geographical terms
def geographical_terms():
    print('Extracting geographical terms...')
    # page 1
    geographical_pages = {} # key: term, value: wikipage term, for example: apple: apple (brand)

    wiki_url = "https://de.wikipedia.org/wiki/Geographie_Europas"
    response = requests.get(wiki_url)
    assert response.status_code == 200

    soup = BeautifulSoup(response.text, 'html.parser')
    soup = soup.find_all('ol')[0:-1]
    for section in soup:
        for item in section.find_all('li'):
            term = item.find('a')['title']
            if term.split(' ') == '(Seite nicht vorhanden)': # items whose pages are empty
                continue
            page = wiki_wiki.page(term)
            if page.exists() == False: # if the term does not lead to readable page for wikiapi
                continue
            new_term = term.split(' (')[0]
            geographical_pages[new_term] = term

    # page 2
    wiki_url = "https://de.wikipedia.org/wiki/Liste_von_W%C3%BCsten_in_Afrika"
    response = requests.get(wiki_url)
    assert response.status_code == 200
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    for item in table.find_all('tr'):
        try:
            term = item.find_all('a')[0]['title']
        except:
            continue
        page = wiki_wiki.page(term)
        if page.exists() == False: # if the term does not lead to readable page for wikiapi
            continue
        geographical_pages[term] = term

    # page 3
    wiki_url = 'https://de.wikipedia.org/wiki/Liste_deutscher_Inseln'
    response = requests.get(wiki_url)
    assert response.status_code == 200
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    for row in table.find_all('tr')[1:]:
        term = row.find_all('a')[0]['title']
        page = wiki_wiki.page(term)
        if page.exists() == False: # if the term does not lead to readable page for wikiapi
            continue
        geographical_pages[term] = term

    # page 4
    wiki_url = 'https://de.wikipedia.org/wiki/Liste_der_Nationalparks'
    response = requests.get(wiki_url)
    assert response.status_code == 200
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find_all('table')[-3:-1]
    for t in table:
        for row in t.find_all('tr')[1:]:
            try:
                term = row.find_all('td')[4].find('a')['title']
            except:
                continue
            page = wiki_wiki.page(term)
            if page.exists() == False: # if the term does not lead to readable page for wikiapi
                continue
            geographical_pages[term] = term
            
    # page 5
    wiki_url = 'https://de.wikipedia.org/wiki/Liste_von_Karstlandschaften'
    response = requests.get(wiki_url)
    assert response.status_code == 200
    soup = BeautifulSoup(response.text, 'html.parser')
    uls = soup.find_all('ul')[6:25] # 太多了 目前 218 個 可以減少
    for ul in uls:
        rows = ul.find_all('a')
        for row in rows:
            term = row['title']
            page = wiki_wiki.page(term)
            if page.exists() == False: # if the term does not lead to readable page for wikiapi
                continue
            new_term = term.split(' (')[0]
            geographical_pages[new_term] = term

    return geographical_pages
# extract geopolitical terms
def geopolitical_terms():
    print('Extracting geopolitical terms...')

    geopo_pages = {}
    # page 1
    wiki_url = 'https://de.wikipedia.org/wiki/Liste_der_Hauptst%C3%A4dte_Europas'
    response = requests.get(wiki_url)
    assert response.status_code == 200
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    terms = []
    for row in table.find_all('tr')[1:]:
        capital = row.find_all('td')[0].find('a')['title']
        country = row.find_all('td')[4].find('a')['title']
        terms.append(capital)
        terms.append(country)
    for term in terms:
        page = wiki_wiki.page(term)
        if page.exists() == False: # if the term does not lead to readable page for wikiapi
            continue
        new_term = term.split(' (')[0]
        geopo_pages[new_term] = term
    
    #page 2
    wiki_url = 'https://de.wikipedia.org/wiki/Liste_der_gr%C3%B6%C3%9Ften_St%C3%A4dte_der_Welt_(historisch)'
    response = requests.get(wiki_url)
    assert response.status_code == 200
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find_all('table')[-1]
    terms = []
    for row in table.find_all('tr')[2:]:
        capital = row.find_all('td')[1].find('a')['title']
        country = row.find_all('td')[2].find('a')['title']
        terms.append(capital)
        terms.append(country)
    for term in terms:
        page = wiki_wiki.page(term)
        if page.exists() == False: # if the term does not lead to readable page for wikiapi
            continue
        new_term = term.split(' (')[0]
        geopo_pages[new_term] = term
    
    #page 3
    wiki_url = 'https://de.wikipedia.org/wiki/Liste_der_Hauptst%C3%A4dte_der_Erde'
    response = requests.get(wiki_url)
    assert response.status_code == 200
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    terms = []
    for row in table.find_all('tr')[1:]:
        try:
            capital = row.find_all('td')[1].find('a')['title']
            country = row.find_all('td')[0].find('a')['title']
        except:
            continue
        terms.append(capital)
        terms.append(country)
    for term in terms:
        page = wiki_wiki.page(term)
        if page.exists() == False: # if the term does not lead to readable page for wikiapi
            continue
        new_term = term.split(' (')[0]
        geopo_pages[new_term] = term
        if len(geopo_pages) >= 200:
            break
    return geopo_pages
# extract structural terms
def structural_terms():
    print('Extracting structural terms...')
    structural_pages = {}
    #page 1
    wiki_url = 'https://de.wikipedia.org/wiki/Weltwunder'
    response = requests.get(wiki_url)
    assert response.status_code == 200
    soup = BeautifulSoup(response.text, 'html.parser')
    for ul in soup.find_all('ul')[3:5]:
        lis = ul.find_all('li')
        for li in lis:
            term = li.find('a')['title']
            page = wiki_wiki.page(term)
            if page.exists() == False: # if the term does not lead to readable page for wikiapi
                continue
            new_term = term.split(' (')[0]
            structural_pages[new_term] = term
    
    #page 2
    wiki_url = 'https://de.wikipedia.org/wiki/Liste_rekonstruierter_Bauwerke_in_Deutschland'
    response = requests.get(wiki_url)
    response.status_code == 200
    soup = BeautifulSoup(response.text, 'html.parser')
    for table in soup.find_all('table'):
        for row in table.find_all('tr')[1:]:
            term = row.find_all('td')[0].find('a')['title']
            page = wiki_wiki.page(term)
            if page.exists() == False: # if the term does not lead to readable page for wikiapi
                continue
            new_term = term.split(' (')[0]
            structural_pages[new_term] = term
            
    return structural_pages

def main():
    # perform the extraction
    result = {}
    result['GEO'] = geographical_terms()
    result['GPE'] = geopolitical_terms()
    result['STRL'] = structural_terms()

    # do some report
    print()
    print(f'Found {len(result["GEO"])} GEO terms.')
    print(f'Found {len(result["GPE"])} GPE terms.')
    print(f'Found {len(result["STRL"])} STRL terms.')

    # save the found terms
    print()
    print(f'Found terms saved to "new_entity_terms.json"')
    with open('new_entity_terms.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent = 4)

if __name__ == "__main__":
    main()
