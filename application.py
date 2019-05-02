#
# Author: Abhay Singh
# UTA ID: 1********9
# This a an application written in python for performing text search ,recommendation and classification.
#


import re
import random
import time
import math
import csv
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import PorterStemmer
from collections import defaultdict
from flask import Flask , render_template , request
import pickle


idfdictionary_global = {}
tfidfquerytermsdictionary_global = {}
documentindexedtfidfscoreofcorpus_global = {}
textual_reviews = {}
idfdictionary_global = {}
textual_reviews = {}
documentCount = 0
porter = PorterStemmer()
pagesize = 10
next = 0
newsdata_global={}

application = Flask(__name__)

'''
    Description : Function storingreviewsindictionary reads the csv file line by line and updates the dictionary
                  with line number as key and value as description of the reviews. This gives the application
                  flexibility to use the in memory object rather then referring to csv file again and again
    Input       : None
    Output      : textual_reviews. This is a dictionary data-structure and it stores document index as the key and
                  text reviews as the value. This dictionary is kept global so that it can be accessed throughout the
                  application.
    
        
'''


def storingreviewsindictionary():
    global textual_reviews
    dictionary = {}
    count = 0
    with open('news_summary.csv', encoding="ISO-8859-1") as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            textual_reviews.update({count: row[1]})
            count = count + 1

    # r = json.dumps(textual_reviews)
    # with open("textual_reviews.json", 'w') as outfile:
    #     json.dump(r, outfile)

    pickle_out = open("textual_reviews.pickle", "wb")
    pickle.dump(textual_reviews, pickle_out)
    pickle_out.close()

    return textual_reviews


'''
    Description : Function parsingDescriptionAndStoringInDictionary reads the global text_reviews dictionary and 
                  by iterating through the keys it performs cleaning of the values in stages as mentioned below.
                  
                  stage 1: Removes the unwanted characters from the review by using a custom defined regular
                           expression. Textual review gets filtered.
                  stage 2: Converting the description from stage 1 to lowercase and removing stop words from
                           reviews using stopwords from nltk package 
                  stage 3: Converting the words present in the reviews to its root word using stemming.
                  
                  Entire data is stored in dictionary data in form of document index and value as the 
                  filtered value of review obtained from stages 1,2 and 3
                  
    Input       : None
    Output      : dictionarydata


'''

def parsingDescriptionAndStoringInDictionary():
    global textual_reviews
    dictionaryData={}
    documentCount = 0

    for key in  textual_reviews:
        stringValue=textual_reviews[key]

        # using RE to remove special characters in the string
       # print(stringValue)
        regexFreeData = re.sub(r"[-()\"#/@;:<>{}`''+=~|.!?,''[]", "", stringValue)

        # Splitting the description column based on space
        valueArray = re.split('\s+', regexFreeData.lower())

        # removing the stopwords from the array
        stop_words = set(stopwords.words('english'))

        #updatedValueArray = removeStopWordsFromDescription(valueArray)
        updatedValueArray = [w for w in valueArray if not w in stop_words]

        #Performing lematization
        dictionaryData[documentCount]=performLematization(updatedValueArray)
        documentCount=documentCount + 1

    return dictionaryData,documentCount




@application.route('/searchengine')
def searchenginemain():
    return render_template('index.html')


@application.route('/recommender')
def recommender():
    return render_template('recommendation.html')


@application.route('/classification')
def searchengine():
    return render_template('classification.html')


@application.route('/')
def index():
    return render_template('main.html')


'''
    Description : This is the main routing function invoked by the container when /searchreviews
    global idfdictionary_global : This aa global variable and it is initialised on the first reque
    global tfidfquerytermsdictionary_global
    global documentindexedtfidfscoreofcorpus_global
    global idfdictionary_global
    global textual_reviews
'''

@application.route('/searchreviews')
def searchreviews():
    # Get the user query from the request
    searchquery =request.args.get('query')
    searchquery1 = request.args.get('recommendation')

    global idfdictionary_global
    global documentindexedtfidfscoreofcorpus_global
    global idfdictionary_global
    global textual_reviews

    if len(documentindexedtfidfscoreofcorpus_global) == 0:

        start = time.time()
        # exists = os.path.isfile('documentindexedtfidfscoreofcorpus.json')
        # if exists:
        #     with open("documentindexedtfidfscoreofcorpus.json") as F:
        #         json_data = json.loads(F.read())
        #         documentindexedtfidfscoreofcorpus_global = ast.literal_eval(json_data)
        # end = time.time()
        pickle_in = open("documentindexedtfidfscoreofcorpus.pickle", "rb")
        documentindexedtfidfscoreofcorpus_global = pickle.load(pickle_in)

    if len(idfdictionary_global) == 0:

        # exists = os.path.isfile('idfdictionary.json')
        # if exists:
        #     with open("idfdictionary.json") as F:
        #         json_data = json.loads(F.read())
        #         idfdictionary_global = ast.literal_eval(json_data)

        pickle_in = open("idfdictionary.pickle", "rb")
        idfdictionary_global = pickle.load(pickle_in)


    if len(textual_reviews) == 0:

        # exists = os.path.isfile('textual_reviews.json')
        # if exists:
        #     with open("textual_reviews.json") as F:
        #         json_data = json.loads(F.read())
        #         textual_reviews = ast.literal_eval(json_data)
        # end = time.time()

        pickle_in = open("textual_reviews.pickle", "rb")
        textual_reviews = pickle.load(pickle_in)







    print( len(idfdictionary_global) , len(documentindexedtfidfscoreofcorpus_global) )

    if len(idfdictionary_global) > 0 and len(documentindexedtfidfscoreofcorpus_global) > 0:
        print("in the if block    ")

        if searchquery != None:
            tfidfquerytermsdictionary = convertQueryIntoDictionary(searchquery, idfdictionary_global)
        if searchquery1 != None:
            tfidfquerytermsdictionary = convertQueryIntoDictionary(searchquery1, idfdictionary_global)

        cosineresult = calculateCosineSimilarityBetweebQueryandDocuments(tfidfquerytermsdictionary,
                                                                         documentindexedtfidfscoreofcorpus_global)
    else:

        # Building the entire document indexing
        #dbquery="select REVIEWS from AMAZONPHONEREV"

        # Connect to IBM Cloud and fetch the Data
        storingreviewsindictionary()

        # Converting the fetched data into dictionary
        dictionaryData, documentCount = parsingDescriptionAndStoringInDictionary()

        # Converting
        validtermfrequency = creatingVectorRepresentationOfDocument(dictionaryData)
        idfdictionary = calculatedocumentfrequency(documentCount)

        idfdictionary_global = idfdictionary
        documentcorpusindexedtfidf = computeTfIDFOfTheCorpus(idfdictionary, validtermfrequency)
        documentindexedtfidfscoreofcorpus_global = documentcorpusindexedtfidf
        tfidfquerytermsdictionary = convertQueryIntoDictionary(searchquery, idfdictionary)
        tfidfquerytermsdictionary_global = tfidfquerytermsdictionary

        cosineresult = calculateCosineSimilarityBetweebQueryandDocuments(tfidfquerytermsdictionary,
                                                                         documentcorpusindexedtfidf)

    if searchquery !=None:
        return render_template('tedresult.html',results=cosineresult , searchquery=searchquery ,tfidfscore=tfidfquerytermsdictionary)
    if searchquery1!=None:
        return render_template('recommendationresult1.html', results=cosineresult ,recommedationfor = searchquery1)


def performLematization(wordtokens):
    wordinflectedlist=[]
    for word in wordtokens:
        wordinflectedlist.append(porter.stem(word))


    return wordinflectedlist

'''
    Description : creatingVectorRepresentationOfDocument creates a term frequency dictionary which is normalised.
    Input       : dictionaryData containing terms and its frequency in a document.
    Output      : documentTermFrequencyDictionary contains document index as key and normalised term frequency as value
                  for each document.
'''

def creatingVectorRepresentationOfDocument(dictionaryData):
    #converting the query in the dictionary format
    documentTermFrequencyDictionary ={}

    for i in dictionaryData:
        arrayData = dictionaryData[i]    # one row of data in array form
        dict1 = dict.fromkeys(list(arrayData), 0)

        for j in range(len(arrayData)):
            dict1[arrayData[j]] += 1
        # this ds has the document number as D2543 : Value as a dictionary which has the termfrequency count of the terms

        normalisedTermFrequencyDictionary = {}
        for n in dict1:

            normalisedTermFrequencyDictionary.update({n:float(dict1[n] /len(arrayData))})

        documentTermFrequencyDictionary.update({i: normalisedTermFrequencyDictionary})

        #Normalising the term count with respect to the number of valid terms in each document

    return documentTermFrequencyDictionary



'''
    Description : calculatedocumentfrequency calculates the inverted document frequency for each term present in each document.
                  Each row read from the csv file is returned as a list of strings .TextBlob is a Python (2 and 3) library for 
                  processing textual data.It provides a simple API for diving into common natural language processing (NLP) tasks
                  such as part-of-speech tagging,noun phrase extraction, sentiment analysis, classification, translation, and more.
                  After applying the textblob to list of terms , it is checked for presence of stop words. Later the resultant list
                  of words goes through stemming (to convert it into its root form). After this series of operation is completed
                  i.e is foreach term of every document we compute the number of documents in which the terms occur. Then we calculate
                  the idf score for each term.
    Input         : Document count (100K in this case)
    Output        : idfDictionary , containing terms and its IDF score. 
'''
def calculatedocumentfrequency(documentCount):
    idfDictionary = {}
    ps = PorterStemmer()
    stop_word = set(stopwords.words('english'))
    word_dict = defaultdict(set)
    with open('./news_summary.csv',encoding="ISO-8859-1") as input:
        oops = csv.reader(input)
        for count, i in enumerate(oops):
            for insert in set(TextBlob(i[1].lower()).words) - stop_word:
                word_dict[ps.stem(insert)].add(count)
    a = dict(word_dict.items())
    idfDictionary = {key: (1 + math.log(documentCount / len(a[key]))) for key in a.keys()}

    endtime = time.time()

    # Writing the idf dictionary in csv file.

    s = time.time()

    # r = json.dumps(idfDictionary)
    # with open("idfDictionary.json", 'w') as outfile:
    #     json.dump(r, outfile)

    pickle_out = open("idfDictionary.pickle", "wb")
    pickle.dump(idfDictionary, pickle_out)
    pickle_out.close()

    return idfDictionary


'''
    Description : computeTfIDFOfTheCorpus caclualtes the TF * IDF score of each term present in the document.
                  This calculation repeats for entire set of documents.

    Input       : idfdictionary contains the inverted document frequency  for each term of a document.
                  documentTermFrequencyDictionary: Contains the term frequency for each document

    Output      : documentindexedtfidfscoreofcorpus contained document number as index and value as
'''

def computeTfIDFOfTheCorpus(idfdictionary , documentTermFrequencyDictionary):

    documentindexedtfidfscoreofcorpus = {}
    docindex=0



    for docid in documentTermFrequencyDictionary.keys():
        tfidfscoreofcorpus = {}
       # print("iterating for document {}".format(docid))
        documentDictionary = documentTermFrequencyDictionary[docid]
       # print("document obtained is {}".format(documentDictionary))
        for terms in documentDictionary.keys():
            if terms in idfdictionary:
                tfidfscoreofcorpus.update({terms : (documentDictionary[terms] * idfdictionary[terms])})

       # print(docindex,tfidfscoreofcorpus)
        documentindexedtfidfscoreofcorpus.update({docindex: tfidfscoreofcorpus})
        docindex = docindex + 1


    # r = json.dumps(documentindexedtfidfscoreofcorpus)
    # with open("documentindexedtfidfscoreofcorpus.json", 'w') as outfile:
    #     json.dump(r, outfile)
    #

    pickle_out = open("documentindexedtfidfscoreofcorpus.pickle", "wb")
    pickle.dump(documentindexedtfidfscoreofcorpus, pickle_out)
    pickle_out.close()

    return documentindexedtfidfscoreofcorpus


'''
    Description :convertQueryIntoDictionary manipulates the search query initiated by the end user. First a regular
                 expression is used to filter out unwanted characters. Then the resultant string is tokenised and s
                 stored in 'arrayQuery'. Then stop words are removed and lemmatisation is performed on the query tokens
                 Then there is calculation for term frequency of the tokens present in the query . Then there is
                 calculation for tf * idf score of the query terms

    Input       : idfdictionary ,Contains the inverted document frequency score for each term of a document.
                  query ,User search query 
    Output      : tfidfquerytermsdictionary, contains the TF * IDF score for each terms in the query
'''


def convertQueryIntoDictionary(query,idfdictionary):
    # Removing the special characters from the input user query


    regexFreeData = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", query)
   # print(regexFreeData.lower())
    arrayQuery = re.split("\s",regexFreeData.lower())

    stop_words = set(stopwords.words('english'))
    # updatedValueArray = removeStopWordsFromDescription(valueArray)
    updatedValueArray = [w for w in arrayQuery if not w in stop_words]

    filteredArrayQuery = performLematization(updatedValueArray)
    queryDictionary = dict.fromkeys(list(filteredArrayQuery), 0)
    for j in range(len(filteredArrayQuery)):
        queryDictionary[filteredArrayQuery[j]] += 1

    # queryDictionary contains terms and its frequency, now normalising the frequency.
    normalisedquerydictionary = {}
    for terms in queryDictionary.keys():
        freq = queryDictionary[terms] / len(queryDictionary)
        normalisedquerydictionary.update({terms:freq})

    #calculating the tf*idf of the query

    tfidfquerytermsdictionary={}
    for queryterms in normalisedquerydictionary.keys():
        if queryterms in idfdictionary.keys():
            tfidfquerytermsdictionary.update({queryterms:(normalisedquerydictionary[queryterms] * idfdictionary[queryterms])})
    return  tfidfquerytermsdictionary





'''
    Description : calculateCosineSimilarityBetweebQueryandDocuments caclulates the vector dot products between tfidfquerytermsdictionary 
                  and documentindexedtfidfscoreofcorpus. Top 10 documents are selected based on their cosine similarity value.
    Input       : tfidfquerytermsdictionary , Dictionary of query terms and their TF * IDF score
                  documentindexedtfidfscoreofcorpus , Dictionary having document index and value as idfdictionary for each document
    Output      : List of 10 documents having the most high cosine similarity value .  
'''

def calculateCosineSimilarityBetweebQueryandDocuments(tfidfquerytermsdictionary , documentindexedtfidfscoreofcorpus):

    cosineresultdictionary={}
    cosineList = []
    docindex = 0
    count = 0
    global textual_reviews

    for documentdictionary in range(len(documentindexedtfidfscoreofcorpus)):

        if docindex == len(documentindexedtfidfscoreofcorpus):
            break
        document = documentindexedtfidfscoreofcorpus[docindex]


        for queryterms in tfidfquerytermsdictionary.keys():

            if queryterms in document:
                cosineList.append([tfidfquerytermsdictionary[queryterms],document[queryterms]])


            else:
                cosineList.append([tfidfquerytermsdictionary[queryterms], 0])
                count = count +1

        # calculate the cosine similarity of the query and document at docindex

            num = 0.0
            totalmag = 0.0
            for product in range(len(cosineList)):
                num = num + cosineList[product][0] * cosineList[product][1]

            querydenominatormag = 0.0
            documentdenominatormag = 0.0

            if num != 0.0:
                for x in range(len(cosineList)):
                    querydenominatormag = querydenominatormag + cosineList[x][0] ** 2
                    documentdenominatormag = documentdenominatormag + cosineList[x][1] ** 2
                totalmag = math.sqrt(querydenominatormag * documentdenominatormag)

                cosineFactor = num / totalmag
                cosineresultdictionary.update({docindex: cosineFactor})


        docindex = docindex + 1
        cosineList.clear()
        count=0
    result = sorted(cosineresultdictionary.items(), key= lambda x: x[1] , reverse=True)

    list=[]
    returnresultcount = 0
    print(result)
    #print(result)
    print("query tfidf score :{}".format(tfidfquerytermsdictionary))
    for k,v in result:

           print("idf score :{}".format(documentindexedtfidfscoreofcorpus[k]))
           d= documentindexedtfidfscoreofcorpus[k]
           r = textual_reviews[k] +"["+ " simlarity score :"+str(v)[0:4]
           scoreofqueryterms =[]
           for q in tfidfquerytermsdictionary:
               terms = ""

               if str(q) in d:
                terms = str(q) +":"+str(d[str(q)])
                scoreofqueryterms.append(terms)

           r= r+  "tf*idf score "+str(scoreofqueryterms)+"]"

           list.append(r)
           returnresultcount = returnresultcount+1
           if returnresultcount == 10:
            break
    return list



@application.route("/pagination")
def pagination():
    global pagesize
    global next
    global newsdata_global

    if len(newsdata_global) == 0:
        print("loading newsdata")
        pickle_in = open("textual_reviews.pickle", "rb")
        newsdata_global = pickle.load(pickle_in)


    print("loaded already")
    start = next
    nextfield = request.args.get('next','')
    previousfield = request.args.get('previous', '')


    if nextfield == 'next':
        next = next + pagesize

    elif previousfield == 'previous':
        next = next - pagesize
        start = start- 2*pagesize
    else:
        print("default return")
        start = 0
        next = 10
        return render_template("recommendationresult.html", news=newsdata_global, start = 0, next=next)

    print(start , next)


    return render_template("recommendationresult.html", news=newsdata_global, start = start, next=next)


'''

    Description : Function readFile reads the input csv data and stores the result in a list
    Input : None
    Output : dataset
'''


def readFile():
    lines = csv.reader(open(r'diabetes.csv'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    # print(dataset)
    return dataset


'''
    Description : Function splitDataset reads the input dataset and splitratio and splits the data in that ratio
                  percentage of the ratio is used as traning data and remaning is considered as test data.dataset is 
                  copied in 'copy' and then using a randomisazed approach , we keep poping the values from copy
                  and storing it into trainset 
    Input : dataset , splitratio
    Output : dataset , copy


'''


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) <= trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    # print(len(trainSet))
    return [trainSet, copy]


'''

    Description : function separateByClass splits the dataset with respect to the last columns , i.e the result column based on 
                  its values 1 and 0 and is stored in separated (list datastructure)
    Input : dataset
    Output : separated


'''


def separateByClass(dataset):
    separated = {}
    classA = []
    classB = []

    for i in range(len(dataset)):
        vector = dataset[i]

        if (vector[-1] == 1.0):
            classA.append(vector)
        else:
            classB.append(vector)
    separated.update({1.0: classA})
    separated.update({0.0: classB})
    return separated


'''
    Description : function clculatestandarddeviation is used to calculate the standard deviation of the input numbers
                  First the mean value is calculated for list of numbers and then the we calculate the variance.
                  Since std deviation is root of variance we calculate  the root of variance
    Input :   numbers
    Output :  math.sqrt(variance) 

'''


def clculatestandarddeviation(numbers):
    avg = calculatemean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


'''
    Description : function calculatemean is used to calculate the average of the input numbers

    Input :   numbers
    Output :  sum(numbers) / float(len(numbers))

'''


def calculatemean(numbers):
    return sum(numbers) / float(len(numbers))


'''
    Description : function summarize calls means and standard deviation for all the  input numbers and 
                  stores it in summaries list of tuple.

    Input :   dataset
    Output :  summaries

'''


def summarize(dataset):
    summaries = [(calculatemean(attribute), clculatestandarddeviation(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


'''
    Description : function summarizeByClass stores the data with respect to class variable .
    Input :   dataset
    Output :  summaries

'''


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


'''
    Description : function calculateProbability calculates the probability using the gaussian distribution .
    Input :   x, mean, stdev
    Output :  gaussian distribution ( (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent )

'''


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


'''
    Description : function calculateClassProbabilities calculates the probability using the gaussian distribution model for each of the 
                  attribute present in the testdata(inputvector) and then stores the value against the class label in dictionary(probabilities)
    Input :   summaries, inputVector
    Output :  probabilities

'''


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


'''
    Description : function predict predicts the value of the input test data i.e input vector and returns 
                  its class label
    Input :   summaries, inputVector
    Output :  bestLabel

'''


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    print("prob", probabilities)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability >= bestProb:
            bestProb = probability
            bestLabel = classValue
    print("best label", bestLabel)
    return bestLabel, probabilities


'''
    Description : function getPredictions simply calls the predict function above and stores
                  result in list. This is called incase where we are calculating the metrics of
                  the classifier model
    Input :   summaries, testSet
    Output :  bestLabel

'''


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result, probabilities = predict(summaries, testSet[i])

        predictions.append(result)
    return predictions


'''
    Description : function getPredictions1 simply calls the predict function above and stores
                  result in list. This is called incase where we are trying to predict result 
                  of some input test data
    Input :   summaries, testSet
    Output :  bestLabel

'''


def getPredictions1(summaries, testSet):
    predictions = []
    probability = []
    for i in range(len(testSet)):
        result, probabilities = predict(summaries, testSet)
        predictions.append(result)
        probability.append(probabilities)
    print(probability)
    return predictions, probability


'''
    Description : function getmetrics calculates the accuracy , precision and recall  
    Input :   summaries, testSet
    Output :  bestLabel

'''


def getmetrics(testSet, predictions):
    correct = 0
    truepostive = 0
    truenegative = 0
    falsepositive = 0
    falsenegative = 0

    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1

        # False negative , result is positive and prediction is negative
        if testSet[x][-1] == 1 and predictions[x] == 0:
            falsenegative += 1

        # True positive, result is positive and prediction is also positive
        if testSet[x][-1] == 1 and predictions[x] == 1:
            truepostive += 1

        # True negative, result is negative and prediction is also negative
        if testSet[x][-1] == 0 and predictions[x] == 0:
            truenegative += 1

        # False positive, result is negative and prediction is positive
        if testSet[x][-1] == 0 and predictions[x] == 1:
            falsepositive += 1

    accuracy = (correct / float(len(testSet))) * 100.0
    precision = (truepostive / (truepostive + falsepositive)) * 100.0
    recall = (truepostive / (truepostive + falsenegative)) * 100.0
    print(accuracy, precision, recall)
    return accuracy, precision, recall


@application.route("/classifier")
def classifier():
    return render_template('classifier.html')


'''

    Description : function classify is the route function invoked by wsgi    
    Input :   summaries, testSet
    Output :  bestLabel

'''


@application.route("/classify")
def classify():
    preganancycount = float(request.args.get('preg'))
    glucose = float(request.args.get('glucose'))
    bloodpressure = float(request.args.get('bp'))
    skin = float(request.args.get('skin'))
    insulin = float(request.args.get('insulin'))
    bmi = float(request.args.get('bmi'))
    diabetes = float(request.args.get('diabetes'))
    age = float(request.args.get('age'))
    splitRatio = 0.70
    dataset = readFile()
    trainingSet, testSet = splitDataset(dataset, splitRatio)

    testSet = [preganancycount, glucose, bloodpressure, skin, insulin, bmi, diabetes, age]

    # prepare model
    summaries = summarizeByClass(trainingSet)

    print("summaries", summaries)

    # test model
    predictions, probalities = getPredictions1(summaries, testSet)

    print("predictions", predictions)

    print("probailities", probalities)

    testresults = None
    if predictions[0] == 1.0:
        testresults = "Test result is positive :-( , You are diabetic"
    elif predictions[0] == 0.0:
        testresults = "Test result is negative :-) , You are not diabetic"

    # accuracy = getAccuracy(testSet, predictions)

    # print('Accuracy: {}'.format(accuracy))
    print("predictions {}".format(predictions))
    return render_template('classifierresult.html', testresult=testresults, prob=probalities,
                           predictions=str(predictions[0]))


'''
    Description : function accuracy is the route function invoked by wsgi    
    Input :   summaries, testSet
    Output :  bestLabel
'''


@application.route("/accuracy")
def accuracy():
    splitRatio = 0.70
    dataset = readFile()
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy, precision, recall = getmetrics(testSet, predictions)
    print("predictions {}".format(predictions))
    return render_template('accuracyresult.html', accuracy=accuracy, precision=precision, recall=recall)



if __name__ == '__main__':
    application.run()