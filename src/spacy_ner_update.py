import tabulate
import spacy
import tqdm
import random
from utils import load
# apply spacy  nlp pipeline
nlp = spacy.load("fr_core_news_md")

sample = "en reculant j'ai embouti ma Renault Clio dans la peugeot 207 du voisin"
doc = nlp(sample)
entities = [(t.text, t.ent_iob_, t.ent_type_) for t in doc]
print(tabulate.tabulate(entities))


# training = [text for text in load() if "clio" in text.lower()][:10]
training = [(['Bonjour', 'Madame', ',', 'Monsieur', '.', "L'", 'année', 'dernière', ',', 'sur', 'vos', 'conseils', ',', 'mon', 'véhicule', 'RENAULT', 'Clio', 'immatriculé', 'AM-534-VP', 'a', 'été', 'assuré', 'par', 'mon', 'fils', ',', 'Thomas', 'TELLIER', '.', 'Désormais', ',', 'celui', '-', 'ci', 'a', 'terminé', 'ses', 'études', 'et', 'réside', 'et', 'travaille', 'à', 'Paris', 'intra', 'muros', '.', 'Il', "n'", 'utilise', 'aucun', 'véhicule', 'personnel', 'pour', 'ses', 'déplacements', '.', 'Je', 'souhaite', 'reprendre', 'à', 'mon', 'nom', "l'", 'assurance', 'de', 'ma', 'Renault', 'Clio', 'puisque', "j'", 'en', 'suis', 'le', 'principal', 'utilisateur', '.', 'Merci', 'de', 'tenir', 'compte', 'de', 'ma', 'demande', 'et', 'de', "m'", 'informer', 'le', 'cas', 'échéant', 'si', 'mon', 'déplacement', 'dans', 'vos', 'bureaux', 'est', 'nécessaire', '.'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'L-LOC', 'O', 'U-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'L-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'L-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']),
(['Bonjour', ',', 'je', 'ne', 'comprends', 'pas', 'votre', 'réponse', 'du', '25', 'avril', 'qui', 'fait', 'référence', 'à', 'un', 'devis', 'du', 'mois', 'de', 'mars', 'sur', 'une', 'clio', 'alors', 'que', 'ma', 'demande', 'de', 'devis', 'concerne', 'une', 'Mercedes', 'GLC', ',', 'merci', 'de', 'revoir', 'vos', 'dossiers', 'et', 'répondre', 'à', 'ma', 'demande', '.'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'U-LOC', 'O', 'O', 'O', 'O', 'O', 'O',
'O', 'O', 'B-LOC', 'L-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']), (['cession', 'du', 'véhicule', 'Clio'], ['O', 'O', 'O', 'U-LOC']), (['Bonjour', 'merci', 'de', 'me', 'faire', 'parvenir', 'au', 'plus', 'tot', 'mon', 'releve', "d'", 'information', 'de', 'la', 'clio', '.', 'Cordialement', ',', 'Bruno', 'Chaplot'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
'O', 'O', 'O', 'O', 'O', 'O', 'L-LOC', 'O', 'O', 'O', 'B-PER', 'L-PER']), (['Bonjour', ',', 'Pouvez', 'vous', 'me', 'donner', 'le', 'prix', 'des', 'options', 'liees', 'au', 'devis', 'realise', 'pour', 'la', 'CLIO', '*', 'Assistance', 'panne', '0', 'km', '*', 'Mise', 'a', 'disposition', "d'", 'un', 'vehicule', 'de', 'pret', '*', 'Vol', 'du', 'contenuMerci'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'U-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']), (['Bonjour', ',', 'Je', 'vais', 'conduire', 'un', 'véhicule', 'RENAULT', 'Clio', 'Immatriculé', '866APY69', 'ce', 'jour', 'Dimanche', '07/10/2019', '.', 'Ce', 'véhicule', 'vient', "d'", 'être', 'assuré', 'au', 'nom', 'de', 'mon', 'fils', 'Sociétaire', 'N', '°', '14238550', '.', 'Je', "n'", 'ai', 'pas', 'réussi', 'à', 'voir', 'si', 'je', 'pouvais', 'me', 'rajouter', 'en', 'deuxième', 'conducteur', 'via', 'son', 'espace', 'assuré', '.', "J'", 'ai', 'donc', 'regardé', 'via', 'le', 'mien', 'mais', 'je', 'ne', 'peux', "m'", 'assurer', 'avec', 'ce', 'véhicule', "qu'", 'à', 'compter', 'de', 'demain', 'alors', 'que', "c'", 'est', "aujourd'hui", 'que', "j'", 'ai', 'besoin', "d'", 'être', 'assurée', 'avec', 'ce', 'véhicule', '.', "J'", 'ai', 'vu', 'sur', 'le', 'contrat', 'de', 'mon', 'fils', ':', 'Prêt', 'de', 'volant', 'mais', 'je', 'ne', 'sais', 'pas', 'ce', 'que', 'cela', 'couvre', 'exactement', '.', 'Je', 'vous', 'remercie', 'donc', 'de', 'prendre', 'en', 'compte', 'ma', 'demande', 'dès', 'à', 'présent', 'Dimanche', '07/10/2019', '11h37', 'afin', 'que', 'je', 'sois', 'assurée', 'aujourd’hui', 'en', 'conduisant', 'ce', 'véhicule', '.', 'Je', 'vous', 'réglerais', 'cette', 'journée', "d'", 'assurance', 'si', 'besoin', '.', 'Merci', 'par', 'avance', ',', 'Cordialement', 'Christine', 'JUNG', 'Tél', ':', '06.67.95.07.03'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'L-LOC', 'O', 'U-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
'O', 'O', 'O', 'O', 'B-PER', 'L-PER', 'O', 'O']),
(['bonjour', '.', 'ma', 'voiture', 'étant', 'au', 'garage', ',', 'je', 'dispose', "d'", 'une', 'voiture', 'de', 'prêt', 'du', 'garage', '.', 'pourriez', 'vous', "m'", 'assurer', 'à', 'la', 'place', 'de', 'la', 'clio', 'pour', 'cette', 'voiture', 'de', 'prêt', ',', 'jusqu’', 'à', 'vendredi', 'soir', '?', "l'", 'immatriculation', 'est', 'CL-266-TQ', ',', "c'", 'est', 'une', 'peugeot', '206', 'mise', 'en', 'circulation', 'le', '10/03/2003', '.', 'merci', 'de', 'me', 'le', 'confirmer', 'par', 'mail', 'ou', 'téléphone', 'svp', '.', 'cordialement', '.', 'mr', 'choplain', '.'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'L-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'U-LOC', 'O', 'O', 'O', 'O', 'B-LOC', 'L-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
'O', 'O', 'O', 'O', 'O', 'B-PER', 'L-PER', 'O']), 
(['Bonsoir', 'Je', 'vous', 'ai', 'contacté', 'tout', 'à', "l'", 'heure', 'pour', 'devis', "d'", 'assurance', 'pour', 'une', 'clio', 'pour', 'le',
'fils', 'de', 'ma', 'femme', 'et', 'je', "n'", 'ai', 'pas', 'reçu', 'devis', 'paf', 'mail', '?', 'Pouvez', 'vous', 'me', "l'", 'envoyer', '?', 'Merci', 'Crdt', '0631243832', '(', 'Autre',
')'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'U-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])]



from spacy.tokens import Doc
from spacy.gold import GoldParse

training_data = []
for tokens, annotation in training:
    doc = Doc(nlp.vocab, words=tokens)
    gold = GoldParse(doc, entities=annotation)
    training_data.append((doc, gold))


for _ in tqdm.tqdm(range(10)):
    random.shuffle(training_data)
    for doc, gold in training_data:
        nlp.update([doc], [gold], drop=0.3)



doc = nlp(sample)
entities = [(t.text, t.ent_iob_, t.ent_type_) for t in doc]
print(tabulate.tabulate(entities))