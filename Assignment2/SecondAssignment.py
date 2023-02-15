import nltk
import benepar
import spacy
benepar.download('benepar_en3')
benepar.download('benepar_fr3')
benepar.download('benepar_it3')
benepar.download('benepar_de3')

nltk.download('punkt')

from nltk import Tree
from supar import Parser
from spacy.tokens import Doc, Span
from benepar.spacy_plugin import BeneparComponent

from spacy.lang.en.examples import sentences 

nlp = spacy.load('en_core_web_sm')
if spacy.__version__.startswith('2'):
    nlp.add_pipe(BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

parsed_string = list(nlp(sentences[3]).sents)[0]._.parse_string

t = Tree.fromstring(parsed_string)
t.pretty_print()

from spacy.lang.fr.examples import sentences 

nlp = spacy.load('fr_core_news_sm')
if spacy.__version__.startswith('2'):
    nlp.add_pipe(BeneparComponent("benepar_fr2"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_fr2"})

parsed_string = list(nlp(sentences[3]).sents)[0]._.parse_string

t = Tree.fromstring(parsed_string)
t.pretty_print()

from spacy.lang.it.examples import sentences 

nlp = spacy.load('it_core_news_sm')
if spacy.__version__.startswith('2'):
    nlp.add_pipe(BeneparComponent("benepar_fr2"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_fr2"})

parsed_string = list(nlp(sentences[1]).sents)[0]._.parse_string

t = Tree.fromstring(parsed_string)
t.pretty_print()

from spacy.lang.de.examples import sentences 

nlp = spacy.load('de_core_news_sm')
if spacy.__version__.startswith('2'):
    nlp.add_pipe(BeneparComponent("benepar_de2"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_de2"})

parsed_string = list(nlp(sentences[2]).sents)[0]._.parse_string
t = Tree.fromstring(parsed_string)
t.pretty_print()