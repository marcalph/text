import tabulate
import spacy
import tqdm
import random
# apply spacy  nlp pipeline
nlp = spacy.load("fr_core_news_md")

from spacy.util import minibatch, compounding

sample = "en reculant j'ai embouti ma Clio 2 dans la peugeot 207 du voisin"
doc = nlp(sample)
entities = [(t.text, t.ent_iob_, t.ent_type_) for t in doc]
print(tabulate.tabulate(entities))


# training = [text for text in load() if "clio" in text.lower()][:10]
training = [
    ('Bonjour, Nous avons vendu la clio, à quelle adresse mail je peux vous envoyer le contrat de cession ? Cordialement Patricia SORTANT ', {"entities": [(29, 33, "VEH")]}), 
    ("Bonjour , pourriez vous aussi corriger l'adresse de mon épouse Mannheimer Séverine qui possède une clio 2 immatriculé BA 523 AK , pour les prochaines attestations d'assurance Je vous en remercie d'avance , je reste a votre disposition pour toute information complémentaire ,Cordialement Famille Mannheimer Alain ", {"entities": [(99, 105, "VEH"), (63, 82, "PER"), (295, 311, "PER")]}),
    ('Bonjour,Je souhaite mettre un terme au contrat qui nous lie concernant la CLIO. Je vous ai envoye le certificat de vente par mail.N?hesitez pas a m contacter.Abdelmajid ', {"entities": [(74, 78, "VEH"), (158, 168, "PER")]}),
    ("Bonjour,Je viens de m'apercevoir que je n'ai pas ete ajoutee en tant que conductrice secondaire sur le contrat d'assurance n? A005 pour notre voiture CLIO ESTATE.Merci de bien vouloir proceder a la modification du contrat.Cordialement,Madame Rebecca MONTIGNY ", {"entities": [(150, 161, "VEH"), (242,258, "PER")]}),
    ("Bonjour, je me sépare de mon véhicule CLIO 2 assuré chez vous sous le contrat réf N° de contrat A005. Sans le remplacer. Cette cession aura lieu le jeudi 30 ami prochain. Je vous remercie donc de bien vouloir résilier le contrat d'assurance propre à ce véhicule dés le 31 mai 2019. Vous remerciant de votre action et de bien vouloir me confirmer cette résiliation. Je suis à votre disposition pour vous faire parvenir par mail les documents attestant de cette vente. Cordialement Alain Moisson ", {"entities": [(38, 44, "VEH"), (480, 493, "PER")]}),
    ('Bonjour Je vous informe que je ne donne pas suite pour assurer chez vous ma Renault Clio immatriculée AM-943-QC. Cordialement Mme Ourabah Isabelle ', {"entities": [(76, 88, "VEH"), (130, 146, "PER")]}),
    ('Apres cession de mon vehicule CLIO le 22/01/2015 a 21h30mn, pourriez vous resilier le contrat concernant la Clio et reporter l\'avoir sur la 406?Puis-je vous envoyer avec une piece jointe (format PDF ou jpeg) un "scan" de la declaration de cession de vehicule (cerfa 1375*02), et dans ce cas comment ?Sinon par courrier postal, et dans cas adresse et service concerne ,Sinceres remerciements et salutations.P. Mathieu ', {"entities": [(30, 34, "VEH"), (108, 112, "VEH"), (141,144, "VEH"), (407, 417, "PER")]}),
    ('Je vous prie de bien vouloir noter la vente du véhicule suivant, qui était couvert gratuitement pendant la période de mise en vente : CLIO DT-760-TX Cordialement, J.PASTUREL ', {"entities": [(134, 138, "VEH"), (163, 173, "PER")]}),
    ('Bonjour, Jaimerais assurer un véhicule supplémentaire il est possible de me rappelez ? Il sais donc du devis fait dans mon espace la Clio 4 trophy rs Merci. Cordialement ', {"entities": [(133, 149, "VEH")]}),
    ("bonjour je désire résilier le contrat concernant la clio à partir du 1 mars 2018 mon gendre va reprendre l'assurance en son nom à sa propre assurance quelle démarche dois je faire ? merci de votre réponse salutations ", {"entities": [(52, 56, "VEH")]})]

training_sample = training[-2][0]
print(training_sample)

ner = nlp.get_pipe("ner")
ner.add_label("VEH")
optimizer = nlp.resume_training()
move_names = list(ner.move_names)
print(f"move_names {move_names}")
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
print(f"other pipes {other_pipes}")
with nlp.disable_pipes(*other_pipes):  # only train NER
    sizes = compounding(1.0, 4.0, 1.001)
    # batch up the examples using spaCy's minibatch
    for itn in range(100):
        random.shuffle(training)
        batches = minibatch(training, size=sizes)
        losses = {}
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
        print("Losses", losses)


print("retest sample\n")
doc = nlp(sample)
entities = [(t.text, t.ent_iob_, t.ent_type_) for t in doc]
print(tabulate.tabulate(entities))


print("retest training sample\n")
doc = nlp(training_sample)
entities = [(t.text, t.ent_iob_, t.ent_type_) for t in doc]
print(tabulate.tabulate(entities))









