import math
import numpy as np

class Retrieve:
    def __init__(self,index,termWeighting):
       
        self.index = index
        self.termWeighting = termWeighting
    # get the total number of documents for each collection
    def size(self):
        invertedIndex = self.index
        documentCollection = set([])
        for keys in invertedIndex:
            for key in invertedIndex[keys]:
                documentCollection.add(key)
        return documentCollection
    #get the cosine similarity between two documents
    def similarity(self,query,document):
        docs = {}
       
        for key in document.keys():
            if key in query.keys():
                #calculate the similarity
                docs[key]=(query[key]/document[key])
           
        return sorted(docs, key=docs.get,reverse=True)
    #get the set of documents that have at least one query term 
    def candidateSet(self,query):
        candidateSet = set([])
        invertedIndex = self.index
        # get query term
        for k in query.keys():  
            # if query is in the inverted index
            if  k in invertedIndex:       
                termDocs = invertedIndex[k]
                for key in termDocs:
                    #add the documents that have at least one query term to a set
                    candidateSet.add(key)
        return candidateSet
      
    def forQuery(self,query):
        collectionSize = len(self.size()) 
        invertedIndex=self.index
        # size of each document vector.
        docsToDocVec = {}
        

        #set of documents containg at least on query term
        candidateSet = self.candidateSet(query)
        if self.termWeighting == 'binary':
            #dictionary that holds the size of each  vector
            
            binaryXDocVec = {}
                                   
            #find the size of each binary document vector 
         
            for doc in candidateSet:
                binaryVecWeights = 0
                binaryWeights = 0
                
                #find the binary vector for each query term and document that contains a query term
                for k,v in invertedIndex.items():
                    # if the query term is in the inverted index set its vector to 1 
                    if k in query.keys() and doc in v:           
                        binaryWeights +=1 # sum of binary vectors
                    if doc in v:
                        binaryVecWeights +=1 #sum of document vectors
                # add word and document vectors to a dictionariy for their similarity to be calculated
                docsToDocVec[doc]= np.sqrt(binaryVecWeights)
                binaryXDocVec[doc]=binaryWeights
                        
                # get cosine similarity.
            return self.similarity(binaryXDocVec,docsToDocVec)
        elif self.termWeighting == 'tf':
             #dictionary that holds the size of each  vector
             tfXDocVec = {}
            
                            
                   
             for doc in candidateSet:
                tfVecWeights = 0
                tfWeights = 0
                #find the term freqency vector for each query term and document that contains a query term    
                for k in invertedIndex:
                    #find the size of the document vector
                    if invertedIndex.get(k, False).get(doc, False) != False :
                        tfVecWeights += (invertedIndex.get(k).get(doc))**2 # sum of document vectors
                    if k in query.keys():
                        if invertedIndex.get(k, False).get(doc, False) != False : # check if query term is in the index 
                            if query[k] >1: # check if the number of occurences for the query term is greater than 1
                                tfWeights += (query[k] * invertedIndex.get(k).get(doc) ) # sum the tf vectors
                            else:
                                tfWeights += (invertedIndex.get(k).get(doc) )# sum the tf vectors
                # add term and document vectors to a dictionariy for their similarity to be calculated          
                docsToDocVec[doc]= np.sqrt(tfVecWeights)
                tfXDocVec[doc]=tfWeights

             # get cosine similarity.
             return self.similarity(tfXDocVec,docsToDocVec)
        else:
             #dictionary that holds the size of each  vector
             tfidfXDocVec = {}
                            
             #find the size of each binary document vector           
             for doc in candidateSet:
                tfidfVecWeights = 0
                tfidfWeights = 0
                #find the tfidf vector for each query term and document that contains a query term 
                for k in invertedIndex:
                    # calcualte the inverse document frequency score
                    idf_calculation = collectionSize/len(invertedIndex[k]) 
                    idf = math.log(idf_calculation)
                    
                    if invertedIndex.get(k, False).get(doc, False) != False :
                        tfidfVecWeights +=( (invertedIndex.get(k).get(doc) *idf)**2)# sum of document vectors
                    if k in query.keys():
                        if invertedIndex.get(k, False).get(doc, False) != False :
                            if query[k] >1:
                                tfidfWeights += (query[k] * invertedIndex.get(k).get(doc) *idf ) # sum the tfidf vectors
                            else:
                                tfidfWeights += (invertedIndex.get(k).get(doc) *idf) # sum the tfidf vectors
                # add tfidf and document vectors to a dictionariy for their similarity to be calculated  
                docsToDocVec[doc]= np.sqrt(tfidfVecWeights)
                tfidfXDocVec[doc]=tfidfWeights
                # get cosine similarity.
             return self.similarity(tfidfXDocVec,docsToDocVec)