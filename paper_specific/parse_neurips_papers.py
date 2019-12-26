import argparse

import httplib2
from bs4 import BeautifulSoup, SoupStrainer
import pdftotext
import random
import re
import io
from urllib.parse import urljoin
import requests

##################################################
# Create the argument parser
# http://docs.python.org/dev/library/argparse.html
##################################################
parser = argparse.ArgumentParser(description="Parse arguments")
# python paper_specific/parse_neurips_papers.py https://papers.nips.cc/book/advances-in-neural-information-processing-systems-32-2019 --n 1 --link-filters paper
parser.add_argument("main_url", type=str, help="The main URL for a proceedings page")
parser.add_argument("--link-filters", type=str, nargs="*", help="Filter any links so that they must contain this token")
parser.add_argument("--n", type=int, help="Number of links/papers to randomly sample")
args = parser.parse_args()

input_file = args.main_url

http = httplib2.Http()
status, response = http.request(args.main_url)

links = []
for link in BeautifulSoup(response, parse_only=SoupStrainer('a')):
    if link.has_attr('href'):
        href = link["href"]
        links.append(href)

def _filter_func(link):
    filtered_out = False
    for _filter in args.link_filters:
        if _filter not in link:
            filtered_out = True

    return not filtered_out

links = list(filter(_filter_func, links))

links = random.sample(links, args.n)

def _grab_text_for_pdf(url):
    r = requests.get(url)
    f = io.BytesIO(r.content)
    pdf = pdftotext.PDF(f)
    return "".join(pdf)


compute_terms = ["flop", "fpo", "gpu", "cpu", "hours", "nvidia", "intel", "pflops", "flops", "fpos", "gpu-hours", "cpu-hours", "cpu-time", "gpu-time", "multiply-add", "madd", "tpu"]
energy_terms = ["watt", "kWh", "joule", "joules", "wh", "kwhs", "watts", "rapl"]
carbon_terms = ["co2", "carbon", "emissions"]

term_counts = {
    "energy": [],
    "carbon": [],
    "compute": []
}

def search(target, text, context=6):
    # It's easier to use re.findall to split the string, 
    # as we get rid of the punctuation
    words = re.findall(r'\w+', text)

    matches = (i for (i,w) in enumerate(words) if w.lower() == target)
    for index in matches:
        if index < context //2:
            yield words[0:context+1]
        elif index > len(words) - context//2 - 1:
            yield words[-(context+1):]
        else:
            yield words[index - context//2:index + context//2 + 1]

for link in links:
    #resolve relative urls
    link = urljoin(args.main_url, link + ".pdf")
    print(link)
    text = _grab_text_for_pdf(link).lower()

    for term_list, term_count_index in [(compute_terms, "compute"), (energy_terms, "energy"), (carbon_terms, "carbon")]:
        for term in term_list:
            terms = list(search(term, text))
            if len(terms) > 0:
                print(terms)
                term_counts[term_count_index].append((link, term, " ".join(terms[0])))

for term, counts in term_counts.items():
    print(term)
    for triplets in counts:
        print(",".join(triplets))