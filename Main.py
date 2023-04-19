from ast import Return
from unittest import result
import nltk
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
from numpy.core import records
import scipy.sparse as sp
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from flask import Flask,render_template,request
import os
from flask import request
from flask import Flask, render_template
from flask_table import create_table, Col, Table, Col, LinkCol, ButtonCol
from flask_bootstrap import Bootstrap
import pandas as pd
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import operator
from collections import OrderedDict
app = Flask(__name__)

@app.route("/")
def loginPage():

    return render_template("home.html")

#Main program
@app.route("/main",methods=["POST"])
def main():
    
    global poi,y_score
    poi=request.form["experience"]
    global connection,cursor
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='gadireddy',
                                             user='gadireddy',
                                             password='Ipzkb1jCXJSSCM37sOSJ')
        cursor = connection.cursor()
        pi1=[]
        pj2=[]
        ci1=[]
        rfi1=[]
        candidatepaperlis2=[]
        candidatepaperdic1={}
        candidatepaperdic2={}
        colabfun=[]
        contentfun=[]
        data=[]
        resultlist=[]
        ci1,rfi1,poi,title,abstract=extract_ci_rfi(poi)
        pi1,pj1=extract_pi(ci1)
        pj1,pj2=extract_pi(rfi1)
        candidatepaper,pk=candidate(pi1,pj2,ci1,rfi1)
        candidatepaperlis2,candidatepaperdic1, candidatepaperdic2=candidate_paper(poi,title,abstract,candidatepaper,rfi1,ci1)
        colabfun,z=collabirative(candidatepaperdic1,candidatepaperdic2)
        contentfun=content(candidatepaperlis2)
        contentfun1=content1(candidatepaperlis2)
        rec=get_complements(colabfun,poi,len(contentfun))
        rec1=get_complements(colabfun,poi,len(contentfun1))
        hybridrec=hybrid(rec,contentfun)
        hybridrec1=hybrid(rec1,contentfun1)
        resultlist=list(hybridrec.keys())
        resultlist1=list(hybridrec1.keys())
        x=resultlist[1:11]
        x1=resultlist1[1:11]      
        unique = list(set(x + x1))
        final=random.sample(unique, len(unique))
        data=database(final)
        return render_template("output.html", result_own =  data ) 
        

        
    except mysql.connector.Error as error:
        print("Failed to get record from database: {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
def rating_list(y_true):
    lis=[]
    
    temp=f"select rating from Rating where querykey='{poi}' "
    cursor.execute(temp)
    records=cursor.fetchall()
    if len(records) < 1:
        lis=[0]*len(y_true)
    else:
        for row in records:
            ratingvalue=row[0]
            lis.append(ratingvalue)
    return lis


#insert rating of the user into database
@app.route("/data",methods=["POST"])
def ratings():
    global poi
    lis =[]
    for val in request.form:
        lis.append(val)
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='Name',
                                             user='Name',
                                             password='Password')
        cursor = connection.cursor(buffered=True)
        for val in lis:
            val1 = request.form.getlist(val)
            print("val1",val1)
            key=val1[0]
            rating1=int(val1[1])
            temp=f"select rating from Rating where querykey='{poi}' and recommenderkey='{key}'"
            cursor.execute(temp)
            rating2=cursor.fetchone()
            if rating2 == None:
                rating2=0
                
            if rating2==0:
                ratingfinal=rating1
                sql=f"INSERT INTO Rating (rating,querykey, recommenderkey) VALUES ({rating1},'{poi}','{key}')"
                cursor.execute(sql)
                connection.commit()
            else:
                ratingfinal=(rating1+rating2[0])/2
                sql1=f"INSERT INTO Rating (rating,querykey, recommenderkey) VALUES ({rating1},'{poi}','{key}') ON DUPLICATE KEY UPDATE rating ={ratingfinal}"
                cursor.execute(sql1)
                connection.commit()
        return render_template("home.html")
    except mysql.connector.Error as error:
        print("Failed to get record from database: {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

#Gives the output for the query to display on webapp
def database(resultlist1):
    lis=[] 
    
    for keytobesearched in resultlist1:
        keys,title,abstract,paper_citedby,paper_cite= extract(keytobesearched)   
        dic = {
            "key" : keys,
            "title" : title,
            "abstract" : abstract
        } 
        lis.append(dic)
    return lis

def remove_space(d):
    k=[]
    for i in d:
        j = i.replace(' ','')
        k.append(j)
    return k
#extract papers title,abstract,cite,citedby
def extract(keytobesearched):
    sql_select_Query = f"SELECT * FROM Dblp WHERE keytobesearched="
    temp=sql_select_Query+f"'{keytobesearched}'"
   
    keys_search=" "
    title_search=" "
    abstract_search=" "
    citeby_search=" "
    cite_search=" "   
    cursor.execute(temp)
    records = cursor.fetchall()
    
    for row in records:
        keys_search = row[0]
        title_search = row[1]
        abstract_search = row[2]
        citeby_search = row[3]
        cite_search = row[4]     
    
    return keys_search,title_search,abstract_search,citeby_search,cite_search 
#extract ci and rfi
def extract_ci_rfi(poi):
    citeby=" "
    key_ri=" "
    cite_poi=" "
    title=" "
    abstract=" "
    f=" "
    d=" "
    keytobesearched=poi
    key_ri,title,abstract,citeby,cite_poi = extract(keytobesearched)  
    if citeby is not None :    
        d=citeby[1:-1]
        ci=split(',',d)
    else:
        ci=[]
    if cite_poi is not None :  
        f=cite_poi[1:-1]
        rfi=split(',',f)
    else:
        rfi=[]
    return ci,rfi,poi,title,abstract
#extraction of pi,pj        
def extract_pi(ci):
    pi=[]
    pj=[]
    key_pi=" "
    title=" "
    abs=" "
    l=[]
    k=[]
    citationpaper_cite=" "
    citationpaper_citeby=" "
    ci1=remove_space(ci)
    for keytobesearched in ci1:
       
        key_pi,title,abs,citationpaper_citeby ,citationpaper_cite = extract(keytobesearched)
      
        if citationpaper_citeby is not None :
            f=citationpaper_citeby[1:-1]
            k=split(",",f)
        
        if citationpaper_cite is not None:
            d=citationpaper_cite[1:-1]
            l=split(",",d)
        pi.append(l)
        pj.append(k)
    return pi,pj


def split(delimiters, string, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)
#candidate paper extraction
def candidate(pi,pj,ci,rfi):
    x1 = pi + pj
    pk=empty_conert(x1)
    candidatepaperfinal=[]
    candidatepaper1=[]
    candidatepaper2=[]
    paper_citedby=" "
    key_cp=" "
    pk1=remove_space(pk)
    for keytobesearched in pk1:
        key_cp,title,abs,paper_citedby,paper_cite= extract(keytobesearched)
        if paper_citedby is not None:
            candidatepaper1=append_candidate(keytobesearched,paper_citedby,ci)
            candidatepaper2=append_candidate(keytobesearched,paper_citedby,rfi)
            if candidatepaper1 !=" ":
                candidatepaperfinal.append(candidatepaper1)
            if candidatepaper2 !=" ":
                candidatepaperfinal.append(candidatepaper2)
    return candidatepaperfinal,pk
            
def append_candidate(keytobesearched,paper_citedby,ci):
    key=" "
    papercitedby1=" "
    if  len(paper_citedby) !=0 :
        papercitedby1=paper_citedby[1:-1]
        paperciteby2 = split(',', papercitedby1)
        count1=common_elements(paperciteby2, ci)
    else:
        count1 = 0
    if count1 >= 1:
        key=keytobesearched
    
    return key
#Extraction of poi and candidate papers title,abstract of poi            
def candidate_paper(poi,title,abstract,candidatepaper,rfi,ci):
    mainrfp=[]
    cp_abstract=" "
    cp_title=" "
    cp_cite = " "
    candidatepaperdic={}
    candidatepaperlis1=[]
    candidatepaperdic1={}
    candidatepaperdic2={}
    candidatepaperdic1 ={'key': poi, 'title':title, 'abstract':abstract}
    candidatepaperlis1.append(candidatepaperdic1)
    candidatepaperdic[poi]=rfi
    candidatepaperdic2[poi]=ci
    abstractfinal = []
    titlefinal = []
    cp_keys=" "
    cp_ab=" "
    xcit=[]
    xcit1=[]
    c1=remove_space(candidatepaper)
    for keytobesearched in c1:
        cp_keys ,cp_title ,cp_abstract,cp_citedby,cp_cite = extract(keytobesearched) 
        if cp_cite is not None :  
            d=cp_cite[1:-1]
            xcit=split(",",d)
        if cp_citedby is not None :
            f=cp_citedby[1:-1]
            xcit1=split(",",f)
        candidatepaperdic[keytobesearched]=xcit
        candidatepaperdic2[keytobesearched]=xcit1
        if cp_abstract is not None:
            cp_ab=cp_abstract[1:-1]
        candidatepaperdic1 = {'key': cp_keys, 'title':cp_title, 'abstract':cp_ab}
        candidatepaperlis1.append(candidatepaperdic1)               
        abstractfinal.append(cp_ab)
        titlefinal.append(cp_title)
    return candidatepaperlis1,candidatepaperdic, candidatepaperdic2
#checking common element in two lists
def common_elements(list1, list2):
    count = 0
    for element in list1:
        if element in list2:
            count += 1
    return count
def empty_conert(x):
    z=[j for i in x for j in i]
    p=[k for k in z if k]
    return p
#CFR algorithm
def collabirative(dic1,dic2):
    l,x=dataframe(dic1)
    l1,x1=dataframe(dic2)
    if l1>=l:
        z=x1
    else:
        z=x
    jaccard1=jaccard(dic2)
    jaccard2=jaccard(dic1)
    pos=(0,0)
   
    jaccardz=addAtPos(jaccard1,jaccard2,pos)/2
    JDF = pd.DataFrame(data=jaccardz, index=z,columns=z)
    return JDF,z
def dataframe(dic2):
    df = pd.DataFrame.from_dict(dic2, 'index').stack().reset_index(level=0)
    df2 = df.rename({'level_0': 'key', 0: 'value'}, axis=1) 
    df_pivot = df2.groupby(['key', 'value'])['key'].count()
    df_pivot1 =  df_pivot.unstack().fillna(0)
    return len(df_pivot1),df_pivot1.index 
#adding two matrix or dataframes 
def addAtPos(mat1, mat2, xypos):
    x, y = xypos
    ysize, xsize = mat2.shape
    xmax, ymax = (x + xsize), (y + ysize)
    mat1[y:ymax, x:xmax] += mat2
    return mat1
#jaccard similarity
def jaccard(dic):
    df = pd.DataFrame.from_dict(dic, 'index').stack().reset_index(level=0)
    df2 = df.rename({'level_0': 'key', 0: 'value'}, axis=1) 
    df_pivot = df2.groupby(['key', 'value'])['key'].count()
    data =  df_pivot.unstack().fillna(0)   
    X = np.asmatrix(data.astype(int).values)
    sX = sp.csr_matrix(X) 
    numerator = sX.dot(sX.T)
    ones = np.ones(sX.shape[::-1])
    B = sX.dot(ones)
    denominator = B + B.T - numerator
    jaccard = (numerator / (denominator + 0.00001))
    return jaccard
def get_complements(dataframe,key,n):
    # Returns top n papers
    df=dataframe[key].sort_values(ascending=False)[1:n+1]
    dic=df.to_dict()
    return dic
#CBR algorithm
def generateWordCloudMatrixTitle1(data):
    vectorizer =TfidfVectorizer(lowercase=False,stop_words='english')
    if (str(data['abstract'])!="None"  or str(data['title'])!="None"):
        vectors = vectorizer.fit_transform((data['title']+data['abstract']).values.astype('U'))
    matrixid=pd.DataFrame(vectors.toarray(),columns=vectorizer.get_feature_names())
    return matrixid
def generateWordCloudMatrixTitle(data):
    vectorizer = CountVectorizer(lowercase=False,stop_words='english')
    if (str(data['abstract'])!="None"  or str(data['title'])!="None"):
       vectors = vectorizer.fit_transform((data['title']+data['abstract']).values.astype('U'))
    
    matrixid=pd.DataFrame(vectors.toarray(),columns=vectorizer.get_feature_names())
    return matrixid
def content(data):
    df = pd.DataFrame(data)
    tfmatrix=generateWordCloudMatrixTitle(df)
    similarity_matrix = cosine_similarity(tfmatrix,tfmatrix)
    sim_score=list(enumerate(similarity_matrix[0]))
    sorted_score=sorted(sim_score, key=itemgetter(1), reverse=True)
    simscore=sorted_score[1:5]
    maintitle=[i[0] for i in simscore]
    dict1={}
    for i in range(len(sim_score)):
        dict1[df['key'].iloc[sorted_score[i][0]]]=sorted_score[i][1]
    return dict1
def content1(data):
    df = pd.DataFrame(data)
    tfmatrix=generateWordCloudMatrixTitle1(df)
    similarity_matrix = cosine_similarity(tfmatrix,tfmatrix)
    sim_score=list(enumerate(similarity_matrix[0]))
    sorted_score=sorted(sim_score, key=itemgetter(1), reverse=True)
    simscore=sorted_score[1:5]
    maintitle=[i[0] for i in simscore]
    dict1={}
    for i in range(len(sim_score)):
        dict1[df['key'].iloc[sorted_score[i][0]]]=sorted_score[i][1]
    return dict1
#Hybrid algorithm
def div(my_diction):  
    n = 2  
    for j in my_diction:  
        my_diction[j] = (float)(my_diction[j])/n
    return my_diction
def sorting(dict1):
    sorted_tuples = sorted(dict1.items(), key=operator.itemgetter(1),reverse=True)
    sorted_dict = OrderedDict()
    for k, v in sorted_tuples:
        sorted_dict[k] = v
    return sorted_dict
def hybrid(content_score,collaborative_score):
    contentsim = Counter(content_score)
    colabsim = Counter(collaborative_score)
    sim = (contentsim +colabsim)
    simlar = dict(sim)
    hybrids=div(simlar)
    hybridsim=sorting(hybrids)
    return hybridsim

if __name__ == "__main__":
    print("entered")
    app.run(host='0.0.0.0',port=8080,debug=True)

