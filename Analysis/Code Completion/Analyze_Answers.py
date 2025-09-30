import os
import json
import nltk
import pandas as pd
import sys
global correctnessVar

base_directory = "../../Data/Code Completion/Llama3.1/Results/AnswersZeroShot"
jsonFile = "../../Data/Code Completion/Updated_Test.json"
default_task = "Exact" #Edit
if len(sys.argv) > 3:
    task = sys.argv[3]
    if(task!="Exact" and task!="Edit" and task!="Length"):
        print("Some errors in the tasks,using default configuration")
        task = default_task
    if sys.argv[1]=="Llama":
        base_directory = "../../Data/Code Completion/Llama3.1/Results/Answers"+sys.argv[2]
    elif sys.argv[1]=="Codellama":
        base_directory = "../../Data/Code Completion/Codellama/Results/Answers"+sys.argv[2]
else:
    print("Not enough arguments,using default configuration")
    task=default_task

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
    print(gtFile)
    f = open(gtFile)
    data = json.load(f)
    df = pd.DataFrame(columns=["ID", "PET", "Configuration", "Score", "Length", "Exact Match"])
    for answersPerFile in fileAnswers:
        if(isinstance(answersPerFile, str)):
            id = answersPerFile.split("conf")[0]
        for res in data:
            if (str(res['id'])==id):
                input = res['input'].split(";")[-1]
                gt = res['gt']
                gt = gt.replace(' . ', '.').replace(" ;", ";").replace("<s>", "")
                #print(gt)

        for answers in fileAnswers[answersPerFile][3]:
            conf = fileAnswers[answersPerFile][2]
            pet = answersPerFile.split(conf)[1].replace("_answer.txt", "")
            if(conf=="conf1"):
                score, exact = compareShort(gt, ''.join(answers).rstrip().replace("`", ""),conf)
                length = len(''.join(answers).rstrip().replace("`", ""))
                df.loc[len(df.index)] = [id, pet, "C0", score, length, exact]
            elif(conf=="conf2"):
                score, exact = compareShort(gt, ''.join(answers).rstrip().replace("`", "").replace("<code>","").replace("</code>",""),conf)
                length = len(''.join(answers).rstrip().replace("`", ""))
                df.loc[len(df.index)] = [id, pet, "C1", score, length, exact]
            elif(conf=="conf3"):
                score, exact = compareShort(gt,''.join(answers).rstrip().replace("`", "").replace("<code>","").replace("</code>",""),conf)
                length = len(''.join(answers).rstrip().replace("`", ""))
                df.loc[len(df.index)] = [id, pet, "C2", score, length, exact]
            elif(conf=="conf4"):
                score, exact = compareShort(gt, ''.join(answers).rstrip().replace("`", "").replace("<code>","").replace("</code>",""),conf)
                length = len(''.join(answers).rstrip().replace("`", ""))
                df.loc[len(df.index)] = [id, pet, "C3", score, length, exact]
            elif(conf=="conf5"):
                score, exact = compareShort(gt, ''.join(answers).rstrip().replace("`", "").replace("<code>","").replace("</code>",""),conf)
                length = len(''.join(answers).rstrip().replace("`", ""))
                df.loc[len(df.index)] = [id, pet, "C4", score, length, exact]

    if(task=="Exact"):
        df_pivoted = df.pivot_table(index=["ID", "Configuration"], columns="PET", values=["Exact Match"], aggfunc="sum")
        df_grouped = df_pivoted.groupby("Configuration").sum()#.mean()
    if(task=="Edit"):
        df_pivoted = df.pivot_table(index=["ID", "Configuration"], columns="PET", values=["Score"], aggfunc="mean")
        df_grouped = df_pivoted.groupby("Configuration").mean()
    if(task=="Length"):
        df_pivoted = df.pivot_table(index=["ID", "Configuration"], columns="PET", values=["Length"], aggfunc="mean")
        df_grouped = df_pivoted.groupby("Configuration").mean()
    print(df_grouped)

def compareShort(gt,answer,conf):
    score = nltk.edit_distance(gt.replace(" ",""), answer.replace(" ",""))
    exact_match_counter = 0
    if score < 2:
        exact_match_counter += 1
        #print("Ground Truth: "+gt)
        #print("LLM Answer: "+answer)
    #print("Edit Distance: "+str(score))
    return score, exact_match_counter

comparison(jsonFile, loadAnswers(base_directory))
