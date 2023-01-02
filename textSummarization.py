# Databricks notebook source
#Team Submission for 
# Abhirup Bhattacharya - AXB200027
# Manasi Barhanpurkar - MMB200012




import boto3
from io import StringIO
import pandas as pan
 
!pip install nltk

import re
import math
import nltk
from nltk.corpus import stopwords 
import string


nltk.download('stopwords')
nltk.download('punkt')



class Utils:

    def preprocess(self,data):
        
        stopwordlist = stopwords.words('english')
        
        wordList = []
        counter = 0
        while counter<len(data):
            if data[counter] not in string.punctuation and data[counter] not in stopwordlist:
                wordList.append(data[counter].lower())
            counter += 1
                
        data = ''.join(wordList)
        
        return data


    def vectorFormConvert(self,data1:str,data2:str):
        
        data1_list,data2_list = [],[]
        
        #remove any special characters from the text
        for w in data1.replace("[^0-9A0-Za-z]"," ").split(" "):
            data1_list.append(w.lower())
            
        for w in data2.replace("[^0-9A0-Za-z]"," ").split(" "):
            data2_list.append(w.lower())
            
        #count the unique words
        unique_words = list(set(data1_list + data2_list))
        common_words = list(set(set(data1_list)&set(data2_list)))
        
        vector_1,vector_2 = [0] * len(unique_words),[0] * len(unique_words)
        
        for word in data1_list:
            vector_1[unique_words.index(word)] += 1
            
        for word in data2_list:
            vector_2[unique_words.index(word)] += 1
    
        return common_words, vector_1, vector_2


    def textSimilarityFinder(self,data1:str,data2:str):
        data1 = self.preprocess(data1)
        data2 = self.preprocess(data2)
        
        if data1 == data2:
            return float(0)
        

        common_words, vector_1, vector_2 = self.vectorFormConvert(data1, data2)
    
        numerator = len(common_words)
        
        denominator_1 = math.log(sum(list(vector_1)))
        denominator_2 = math.log(sum(list(vector_2)))

        denominator = denominator_1 + denominator_2
        
        if denominator == 0:
            return float(0)
        else:
            return float(numerator) / denominator







class TextSummarization:
    def __init__(self, aws_acc, 
                 aws_sec, 
                 buck, 
                 inp_f,
                 out_f,
                ):
        self.aws_acc = aws_acc
        self.aws_sec = aws_sec
        self.buck = buck
        self.inp_f = inp_f
        self.out_f = out_f
        self.summ_percent = 0.2

    def readFileToDf(self):
        
        aws_s3 = boto3.client("s3",aws_access_key_id = self.aws_acc,aws_secret_access_key = self.aws_sec)
        #aws_s3 = boto3.client("s3",self.aws_acc,self.aws_sec)

        s3_obj = aws_s3.get_object(Bucket = self.buck,Key = self.inp_f)

        data_body = s3_obj["Body"]
        data_raw = data_body.read()
        data = data_raw.decode('ISO-8859-1')
        
        inp_doc = pan.read_csv(StringIO(data), encoding = 'unicode_escape', delimiter = ',', header=[0])
        print(inp_doc)
        inp_df = spark.createDataFrame(inp_doc)

        return inp_df

    def docToSent(self,textData:str):
        sentenceList = []
        sentenceList = nltk.tokenize.sent_tokenize(textData)

        sentenceList = [sent.replace("[^0-9A0-Za-z]"," ") for sent in sentenceList]

        return sentenceList
    
    def findArticleSummary(self,document_df):
        utils_obj = Utils()
        
        inp_d = document_df.rdd.map(lambda i: (i[0], i[2], i[1])).collect()
        doc_sum = []
        graph = []
        rep_table = []
        
        for d in inp_d:
            doc_id = d[0]
            doc_ttl = d[2]
            doc_sentences = self.docToSent(d[1])
            content = sc.parallelize(doc_sentences)
            cont_arr = content.collect()
            summ_count = int(math.ceil(len(doc_sentences)*self.summ_percent))

            res = []
            summ_str = ""

            for inp_sentence in cont_arr:
                try:
                    sim = sum([x for x in content.map(lambda x: utils_obj.textSimilarityFinder(x, inp_sentence)).collect()]) 
                except:
                    sim = sum([x for x in content.map(lambda x: utils_obj.textSimilarityFinder(x, inp_sentence)).collect()]) 
                res.append((inp_sentence,sim))


            summ_str_rdd = sc.parallelize(res)
            summ_str_sorted = summ_str_rdd.sortBy(lambda v:-v[1])
            summ_str_list = summ_str_sorted.map(lambda v:v[0])
            summ_str_count = summ_str_list.take(summ_count)
            summ_str = ' '.join(summ_str_count)
            
            doc_sum.append((str(doc_id),doc_ttl,summ_str))

            rep_table.append((doc_ttl,len(doc_sentences),summ_count))

            graph.append((doc_ttl,res,summ_str))

        return doc_sum,graph,rep_table


    def out_maker(self,rep_table,mode):
        input_data = "document_id | document_title | document_summary\n"

        for i in range(len(rep_table)):
            rec_data = ""

            if mode in [1,3]:
                print("document_id: ", rep_table[i][0])
                
                doc_title_temp_1 = rep_table[i][1]
                doc_title_temp_2 = doc_title_temp_1.encode("ascii","replace")
                doc_title = doc_title_temp_2.decode("utf-8")
                
                print("document_title: ", doc_title)
                
                doc_sum_temp_1 = rep_table[i][2]
                doc_sum_temp_2 = doc_sum_temp_1.encode("ascii","replace")
                doc_sum = doc_sum_temp_2.decode("utf-8")
                
                print("document_summary: " + doc_sum + "\n")


            if mode in [2,3]:
                rec_data = "|".join(rep_table[i])
                rec_data = rec_data + "\n"

            input_data = input_data + rec_data.encode("ascii","replace").decode()

        if mode in [2,3]:
            
            s3_sess = boto3.Session(self.aws_acc,self.aws_sec).resource('s3')
                                    
            s3Obj = s3_sess.Object(self.buck,self.out_f)

            s = s3Obj.put(Body=input_data)
            print("S3 Upload Successful")






if __name__ == "__main__":
    aws_acc = 'AKIA3SPI3QBVCVWFYEUU'
    aws_sec = 'Gdq87w5bW9148JV/qoUHqY6z6BreaLvhl0kNNAix'
    buck = 'textsummarizationbucket'
    f_input = 'tennis_articles.csv'
    f_output = 'BigData_Summarization_NLP.txt'
    summ_perc = 0.2

    ts_object = TextSummarization(aws_acc, aws_sec, buck, f_input, f_output)

    input_doc_df = ts_object.readFileToDf()
    
    input_doc_df.show()
 



    # article summ
    rep_summ = pan.DataFrame(doc_sum, columns = ["document_id", "document_title", "document_summary"])
    display(rep_summ)



    #Output Modes
    #1 : command line
    #2 : file
    #3 : command line and file

    mode = 2
    ts_object.out_maker(doc_sum, mode)



    #Sentence and summ sentence count specific to document
    rep = pan.DataFrame(rep_table, columns = ["Doducment title", "Count of Sentence", "Count of Summary"])
    display(rep)



# sentence vs weight per document plot
counter = 0
while counter < len(graph):
    print("Document: ", graph[counter][0])
    print("Summ: ", graph[counter][2])
    display(pan.DataFrame(graph[counter][1], columns = ["sentence", "weight"]))
    counter += 1




