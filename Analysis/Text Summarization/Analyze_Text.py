import time
import os
import csv
import nltk
import pandas as pd
import evaluate
import warnings
import sys
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
bleu_scorer = evaluate.load("bleu")
meteor_scorer = evaluate.load("meteor")
bert_scorer = evaluate.load("bertscore")
csv.field_size_limit(1000000)

global correctnessVar

base_directory = "../../Data/Text Summarization/Llama3.1/Results/AnswersZeroShot" #default configuration
gt = "../../Data/Text Summarization/train.csv"
default_task = "rouge" #default configuration
task_length = False

if len(sys.argv) > 3:
    task = sys.argv[3]
    if(task!="rouge"and task!="rougeL" and task!="bleu" and task!="meteor" and task!="bert" and task!="length"):
        task = default_task
    if(task=="length"):
        task = default_task
        task_length = True
    if sys.argv[1]=="Llama":
        base_directory = "../../Data/Text Summarization/Llama3.1/Results/Answers"+sys.argv[2]
    elif sys.argv[1]=="Codellama":
        base_directory = "../../Data/Text Summarization/Codellama/Results/Answers"+sys.argv[2]
else:
    print("Not enough arguments, default configuration")

def edit_distance(gt,answer):
    score = nltk.edit_distance(gt,answer)
    return score

def rouge(gt,answer):
    scores = scorer.score(answer,gt)
    fmeasure = scores['rouge1'][2]#0 precision / 1 recall / 2 fmeasure
    return fmeasure

def rougeL(gt,answer):
    scores = scorer.score(answer,gt)
    fmeasure = scores['rougeL'][2]
    return fmeasure

def bleu(gt,answer):
    scores = bleu_scorer.compute(predictions=[answer], references=[gt])
    result = scores['bleu']
    return result

def meteor(gt,answer):
    scores = meteor_scorer.compute(predictions=[answer], references=[gt])
    result = scores['meteor']
    return result

def bert_score(gt,answer):
    scores = bert_scorer.compute(predictions=[answer],references=[gt],lang="en")
    f1 = scores['f1'][0]
    return f1

metrics_functions = {'rouge':rouge,'rougeL':rougeL,'bleu':bleu,'meteor':meteor,'bert':bert_score}

def loadAnswers(basePath):
    fileAnswers = {}
    for file in os.listdir(basePath):
        answers = []
        data = open(basePath + '/' + file, 'r')
        answer = []
        for line in data.readlines():

            if(">>>Start LLM Answer<<<" in line):
                answer.append(line)
            if(">>>End LLM Answer<<<" in line):
                answer.pop(0)
                answers.append(answer)
                answer = []
            if(">>>Start LLM Answer<<<" not in line and ">>>End LLM Answer<<<" not in line):
                answer.append(line)
        idSnippet = file.split("conf")[0]
        PET = file.split("conf")[1].split("_")[0][1:]
        configuration = "conf"+file.split("conf")[1].split("_")[0][:1]
        fileAnswers[file] = [idSnippet, PET, configuration, answers]
    return fileAnswers

def comparison(gtFile,fileAnswers):
    file = open(gtFile)
    #data = csv.reader(file)
    #next(data)
    data = pd.read_csv(file)
    df = pd.DataFrame(columns=["ID", "PET", "Configuration", "Score", "Length"])
    for answersPerFile in fileAnswers:
        file.seek(0)
        #data = csv.reader(file)
        #next(data)
        if(isinstance(answersPerFile, str)):
            id = answersPerFile.split("conf")[0]
       #id = answersPerFile.split("conf")[0]
        gt = data.loc[data['id'] == int(id)].iloc[0]['description']
        '''
        for res in data:
            #print(id)
            #print(res[0])
            if (str(res[0])==id):
                gt = res[1]
        '''
        for answers in fileAnswers[answersPerFile][3]:
            conf = fileAnswers[answersPerFile][2]
            pet = answersPerFile.split(conf)[1].replace("_answer.txt", "")
            if(conf=="conf1"):
                score = metrics_functions[task](gt, ''.join(answers))
                length = len(''.join(answers).rstrip())
                df.loc[len(df.index)] = [id, pet, "C0", score, length]
            elif(conf=="conf2"):
                score = metrics_functions[task](gt, ''.join(answers))
                length = len(''.join(answers).rstrip())
                df.loc[len(df.index)] = [id, pet, "C1", score, length]
            elif(conf=="conf3"):
                score = metrics_functions[task](gt, ''.join(answers))
                length = len(''.join(answers).rstrip())
                df.loc[len(df.index)] = [id, pet, "C2", score, length]
            elif(conf=="conf4"):
                score = metrics_functions[task](gt, ''.join(answers))
                length = len(''.join(answers).rstrip())
                df.loc[len(df.index)] = [id, pet, "C3", score, length]

    if task_length:
        df_pivoted = df.pivot_table(index=["ID", "Configuration"], columns="PET", values="Length", aggfunc="mean")
        df_grouped = df_pivoted.groupby("Configuration").mean()#.sum
    else:
        df_pivoted = df.pivot_table(index=["ID", "Configuration"], columns="PET", values="Score", aggfunc="mean")
        df_grouped = df_pivoted.groupby("Configuration").mean()#.sum
    print(df_grouped)

comparison(gt, loadAnswers(base_directory))
