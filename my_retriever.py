import math
import numpy as np


class Retrieve:
    def __init__(self,index,termWeighting):
       
        self.index = index
        self.termWeighting = termWeighting
    # get the total number of documents fro each collection
    def size(self):
        invertedIndex = self.index
        documentCollection = set([])
        for keys in invertedIndex:
            for key in invertedIndex[keys]:
                documentCollection.add(key)
        return documentCollection
        
    def similarity(self,query,document):
        docs = {}
        
        for key in document.keys():
            if key in query.keys():
                docs[key]=(query[key]/document[key])
           
        return sorted(docs, key=docs.get,reverse=True)
        
    def candidateSet(self,query):
        candidateSet = set([])
        invertedIndex = self.index
        # get query term
        for k in query.keys():  
            # if query is in the inverted index
            if  k in invertedIndex:       
                # the number of documents that contain the query term
                documentFrequency = len(invertedIndex[k])        
                termDocs = invertedIndex[k]
                for key in termDocs:
                    #add the documents that have at least one query term to a set
                    candidateSet.add(key)
        return candidateSet
      
    def forQuery(self,query):
        collectionSize = len(self.size()) 
        invertedIndex=self.index
        docsToDocVec = {}
        #set of documents containg at least on query term
        candidateSet = self.candidateSet(query)
        if self.termWeighting == 'binary':
         
            binaryXDocVec = {}
                                   
            #find the size of each binary document vector           
            for doc in candidateSet:
                binaryVecWeights = []
                binaryWeights = []
                
                #find the size of each binary document vector
                for k,v in invertedIndex.items():
                    if k in query.keys() and doc in v:           
                        binaryWeights.append(1)
                    if doc in v:
                        binaryVecWeights.append(1**2)
                docsToDocVec[doc]= np.sqrt(np.sum(binaryVecWeights))
                binaryXDocVec[doc]=np.sum(binaryWeights)
                        
                
            return self.similarity(binaryXDocVec,docsToDocVec)
        elif self.termWeighting == 'tf':
             tfXDocVec = {}
                            
             #find the size of each binary document vector           
             for doc in candidateSet:
                tfVecWeights = []
                tfWeights = []
                for k in invertedIndex:
                    
                    if invertedIndex.get(k, False).get(doc, False) != False :
                        tfVecWeights.append((invertedIndex.get(k).get(doc))**2)
                    if k in query.keys():
                        if invertedIndex.get(k, False).get(doc, False) != False :
                            if query[k] >1:
                                tfWeights.append(query[k] * invertedIndex.get(k).get(doc) )
                            else:
                                tfWeights.append(invertedIndex.get(k).get(doc) )
                            
                docsToDocVec[doc]= np.sqrt(np.sum(tfVecWeights))
                tfXDocVec[doc]=np.sum(tfWeights)

                
                        
                        
                
             return self.similarity(tfXDocVec,docsToDocVec)
        else:
            print('tfidf')