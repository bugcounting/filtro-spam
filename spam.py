#! /usr/bin/env python3

import pandas
import os
from sklearn.feature_extraction.text import CountVectorizer


def contenuto_messaggio(path, lingspam=False):
    """Contenuto del messaggio di posta elettronica in file `path`."""
    ## apri file con nome `path`
    with open(path, "r", encoding="latin-1") as fp:
        ## estrai linee di testo in un array
        lines = fp.readlines()
        ## nelle email NON del gruppo lingspam, il contenuto inizia dopo la prima linea vuota
        if not lingspam:
            ## trova la prima occorrenza di una linea vuota
            index_empty = lines.index("\n")
            ## tieni solo le linee del messaggio dopo la linea vuota
            lines = lines[index_empty+1:]
        ## ricomponi le linee in un'unica stringa
        return "".join(lines)


def matrice_occorrenze_termini(paths):
    """Matrice con occorrenze di parole in tutti i file nella lista `paths`.

       Ogni colonna della matrice corrisponde a una parola (termine)
       incontrato almeno una volta nei file.  Ogni riga della matrice
       corrisponde a uno dei file in `paths`.  Il numero che compare
       nella matrice alla riga `r` e alle colonna `c` indica il numero
       di volte che il termine `c` compare nel file `r`.
    """
    ## estrai il contenuto di ogni file in una lista
    messages = [contenuto_messaggio(p) for p in paths]
    ## converti i caratteri in minuscolo, ignora parole comuni, e ignora parole con numeri
    vec = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b")
    ## conta le parole nei messaggi
    X = vec.fit_transform(messages)
    ## costruisci una matrice dove ogni riga è un documento e ogni colonna è una parola
    df = pandas.DataFrame(X.toarray(), columns=vec.get_feature_names(), index=paths)
    ## nota che `columns` si riferisce alle colonne, mentre `index` indica le righe
    return(df)


def tabella_frequenze_termini(paths):
    """Tabella con frequenze dei termini in tutti i file nella lista `paths`.
    """
    occ = matrice_occorrenze_termini(paths)
    ## sostituisci il conteggio di ogni termine con 1 se il termine
    ## compare una o più volte, e con 0 altrimenti
    binary_count = occ.applymap(lambda c: 1 if c > 0 else 0)
    ## la frequenza di un termine è la frazione di file nei quali
    ## compare almeno una volta
    term_frequency = binary_count.sum(axis="index")/len(occ.index)
    ## somma ogni colonna, così da contare quante volte ciascun
    ## termine è comparso (in tutti i file)
    term_occurrence = occ.sum(axis="index")
    ## la densità di un termine è la frazione di tutti i termini (in
    ## qualsiasi file) che sono uguali al termine in questione
    term_density = term_occurrence/sum(term_occurrence)
    data = {"conteggio": term_occurrence,
            "frequenza": term_frequency,
            "densita": term_density}
    tab = pandas.DataFrame(data)
    return(tab)


def conteggio_termini(path):
    """Lista di tutti i termini che compaiono nel file `path`, con il
       numero di occorrenze di ciascun termine.
    """
    tf = tabella_frequenze_termini([path])
    ## della tabella delle frequenze ci interessa solo la colonna "conteggio"
    return(tf["conteggio"])
    

def messaggi_in_cartella(directory):
    """Lista di tutti i file nella cartella `directory` che corrispondono a
       messaggi di posta elettronica.
    """
    ## lista di tutti i file (paths) in `directory`
    paths = os.listdir(directory)
    ## rimuovi i file "cmds" che non sono messaggi di posta elettronica
    ## e completa i vari path con l'indicazione della cartella
    paths = [os.path.join(directory, p) for p in paths if not p.endswith("cmds")]
    return(paths)


def posterior(path, training_data, prior, prob_omessi):
    """Probabilità a posteriori che i termini contenuti nel file `path`
       siano della stessa natura di quelli usati per le statistiche in
       `training_data`.

       prior: probabilità a priori per lo stesso evento di quella a
              posteriori

       prob_omessi: probabilità (piccola) arbitrariamente assegnata a
                    tutti i termini che non si trovano nei dati di
                    training
    """
    ## statistiche sui termini nel file `path`
    frequenze_termini = tabella_frequenze_termini([path])
    ## lista dei termini trovati nel file `path`
    termini = frequenze_termini.index
    ## lista dei termini trovati nei dati di training
    termini_training = training_data.index
    ## lista dei termini trovati sia nel file `path` che nei dati di training
    termini_comuni = [t for t in termini_training if t in termini]
    ## lista di frequenza dei termini comuni nei dati di training
    frequenze_comuni = training_data.loc[termini_comuni]["frequenza"]
    ## prodotto di tutti i valori in `frequenze_comuni`
    likelihood_comuni = frequenze_comuni.product()
    ## `prob_omessi` moltiplicato per se stesso per ogni termine nel
    ## file `path` che non compare nei dati di training
    likelihood_omessi = (prob_omessi ** (len(termini) - len(termini_comuni)))
    ## la likelihood complessiva è il prodotto delle due likelihood parziali
    likelihood = likelihood_comuni * likelihood_omessi
    ## infine il posterior è il prodotto di likelihood e prior
    return(likelihood * prior)


def probabilmente_spam(path, spam_data, ham_data, prior_spam=0.5, prob_omessi=1e-6):
    """Restituisce True se la probabilità a posteriori che il file `path`
       sia un messaggio di spam è maggiore della probabilità a
       posteriori che sia un messaggio di ham.

       spam_data: statistiche (tabella delle frequenze) su messaggi di
                  spam, usate per stimare la likelihood che `path` sia
                  un messaggio di questa natura

       ham_data: statistiche (tabella delle frequenze) su messaggi di
                 ham, usate per stimare la likelihood che `path` sia
                 un messaggio di questa natura

       prior_spam: probabilità a priori che il file `path` sia un
                   messaggio di spam

       prob_omessi: probabilità (tipicamente piccola ma positiva)
                    arbitrariamente assegnata a tutti i termini che
                    non si trovano nei dati di training

    """
    prob_spam = posterior(path, spam_data, prior_spam, prob_omessi)
    prob_ham = posterior(path, ham_data, 1-prior_spam, prob_omessi)
    return prob_spam > prob_ham
