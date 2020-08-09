from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import spacy
spacy.__version__

import spacy.lang.sv.stop_words as sv_stop_words

print("Stop words in SpaCy: ", len(sv_stop_words.STOP_WORDS))
[sw for i, sw in enumerate(sv_stop_words.STOP_WORDS) if i < 5]


contents =  ["""Vägen har inget gemensamt vägnummer, men vimlar av sevärdheter. Sväng av E4 vid Upplands Väsby och strax börjar äventyret vid gamla apoteket i Hammarby där apotekaren mördades 1913. Stanna i Apoteket och få lite i magen.""",
            """En ny jätteundersökning avslöjar vilka bilar som hamnar på verkstaden oftast. Och hur kostsamma de kan bli att äga. Det är smått skrämmande läsning för vissa biltillverkare. Här är listan du INTE kan vara utan om du ska köpa ny eller begagnad bil.""",
            """Mauro Scocco hade egentligen tänkt sluta som artist men en demonproducent och en separation drev honom tillbaka till studion. Som en bonus lyckades han även bota den scenskräck som hållit honom borta från turnélivet i nästan 30 år."""]



import nltk

from nltk.corpus import stopwords
nltk.download('stopwords')

print("Stop words in SpaCy: ", len(stopwords.words('swedish')))
print(stopwords.words('swedish'))