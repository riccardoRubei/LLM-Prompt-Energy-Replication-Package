import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer,pipeline
from codecarbon import EmissionsTracker
import time
import json
import os
import csv
import marko
import copy
from bs4 import BeautifulSoup

csv.field_size_limit(1000000)

def loadModel():

	TOKEN=""
	#model_id = "meta-llama/Meta-Llama-3-8B"
	model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"


	print('Loading Tokenizer')
	tokenizer = AutoTokenizer.from_pretrained(model_id,token=TOKEN)
	tokenizer.pad_token = tokenizer.eos_token

	nf4_config = BitsAndBytesConfig(
	   load_in_4bit=True,
	   bnb_4bit_quant_type="nf4",
	   bnb_4bit_use_double_quant=True,
	   bnb_4bit_compute_dtype=torch.bfloat16
	)

	terminators = [
		tokenizer.eos_token_id,
		tokenizer.convert_tokens_to_ids("<|eot_id|>")
	]
		
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
    	#torch_dtype=torch.bfloat16,
        torch_dtype="auto",
    	device_map="auto",
    	#quantization_config=nf4_config,
    	token=TOKEN
	)

	print('Model loaded')
	return model, tokenizer

	
	
ExperimentsDir = "Experiments"
AnswersDirs = "Answers"
def loadConfigurations(ExperimentsDir,AnswersDirs,textsDone):

	Readmes = []
	ids = []
	maxLength = 3000

	minLength = 1900
	data = open("../GitSum/data/train.csv")
	dataReader = csv.reader(data)
	next(dataReader)

	for row in dataReader:
		#html_Version(row)
		if len(row[2])<maxLength and len(row[2])>4:

			if(str(row[0]) not in textsDone):
				#Readmes.append(entry['input'].replace("<s> ","").replace("</s>","").replace(" . ",".").replace(" .",".").replace(" ;",";"))
				Readmes.append(row[2])
				
				ids.append(row[0])
	print(len(Readmes))
	return Readmes, ids


######TEST SPECIFIC text######
#texts = []
#ids = []
#for entry in data:
#	if(entry['id'] == 127):
#		print(len(entry['input']))
#		texts.append(entry['input'].replace("<s> ","").replace("</s>","").replace(". ","").replace(" ;",";"))
#		ids.append(entry['id'])	

		
def recoverState(ExperimentsDir):
	readmesDone = []
	for file in os.listdir(ExperimentsDir):
		id = file.split("conf")[0]
		readmesDone.append(id)
	return readmesDone

def oneShot(texts,ids,model,tokenizer,AnswersDir,QuestionsDir,ExperimentsDir,limittexts):
	
	user_sample = '''## Introduction

Rucene is a Rust port of the popular Apache Lucene project. Rucene is not a complete application, but rather a code library and API that can easily be used to add full text search capabilities to applications.

## Status

The index searcher part of Rucene has been put into production and has served all search traffics at Zhihu since July, 2018. Development of the index writer part was started in late 2018, and has been put into production to serve real-time searching since May, 2019.

## Documentation

We don't yet have an API documentation for Rucene, but the usage is similar to [Lucene 6.2.1](https://lucene.apache.org/core/6_2_1/).

> **Note:**
>
> We are working on this, but could use more help since it is a massive project.
'''
	
	one_shot_messages_template_base = [
    {
        "role": "system",
        "content": "You are an AI assistant specialized in Github readme summarization. Your task is to summarize the provided readme into a short description. Give only the short description.",
    },
		
	{
		'role': 'user', 
		'content': ""
	},
		
	{
		'role': 'assistant', 
		'content': 'Rust port of Lucene,"Rucene" - Rust implementation of Lucene'
		#'content': 'Example of line to complete the code: byte [ ] b , int off , int len ) throws IOException'
	},
		
    {
        "role": "user",
        "content": None
    }
	]
	
	
	one_shot_messages_template_empty = [
    {
        "role": "system",
        "content": ""
    },
		
	{
		'role': 'user', 
		'content': ""	},
		
	{
		'role': 'assistant', 
		'content': 'Rust port of Lucene,"Rucene" - Rust implementation of Lucene'
	},
		
    {
        "role": "user",
        "content": None
    }
	]
		
	Request_One_Line = "Hi, summarize the provided Github readme into a short description: "
	
	configurations = []
	
	message_html_text = marko.convert(user_sample)
	soup = BeautifulSoup(message_html_text)
	message_plain_text = soup.get_text()
	
	texts = texts[:limittexts]
	for text, id in zip(texts,ids):
		
		#last_position = text.rfind(";")
		#reduced_text = text[:last_position+1]
		#uncompleted_line = text[last_position+1:]
		
		html_readme_prompt = marko.convert(text) 
		soup = BeautifulSoup(html_readme_prompt)
		plain_text_prompt = soup.get_text()
		
		#CONF1 plain
		
		one_shot_messages_conf1 = copy.deepcopy(one_shot_messages_template_base)
		one_shot_messages_conf1[1]['content'] = message_plain_text # system sample message
		one_shot_messages_conf1[-1]['content'] = plain_text_prompt # actual testing prompt
		system_conf1 = one_shot_messages_conf1[0:len(one_shot_messages_conf1)-1]
		message_conf1 = one_shot_messages_conf1[-1]['content']
		input_ids_conf1 = tokenizer.apply_chat_template(one_shot_messages_conf1,add_generation_prompt=True,return_tensors="pt").to(model.device)
		
		#CONF2 markdown
		
		one_shot_messages_conf2 = copy.deepcopy(one_shot_messages_template_base)
		one_shot_messages_conf2[1]['content'] = user_sample # system sample message
		one_shot_messages_conf2[-1]['content'] = text # actual testing prompt
		system_conf2 = one_shot_messages_conf2[0:len(one_shot_messages_conf2)-1]
		message_conf2 = one_shot_messages_conf2[-1]['content']
		input_ids_conf2 = tokenizer.apply_chat_template(one_shot_messages_conf2,add_generation_prompt=True,return_tensors="pt").to(model.device)
		
		#CONF3 html
		
		one_shot_messages_conf3 = copy.deepcopy(one_shot_messages_template_base)
		one_shot_messages_conf3[1]['content'] = message_html_text # system sample message
		one_shot_messages_conf3[-1]['content'] = html_readme_prompt # actual testing prompt
		system_conf3 = one_shot_messages_conf3[0:len(one_shot_messages_conf3)-1]
		message_conf3 = one_shot_messages_conf3[-1]['content']
		input_ids_conf3 = tokenizer.apply_chat_template(one_shot_messages_conf3,add_generation_prompt=True,return_tensors="pt").to(model.device)
		
		#CONF4 no system
		one_shot_messages_template_empty[1]['content'] = user_sample # system sample message
		one_shot_messages_template_empty[-1]['content'] = Request_One_Line + text # actual testing prompt
		system_conf4 = one_shot_messages_template_empty[0:len(one_shot_messages_template_empty)-1]
		message_conf4 = one_shot_messages_template_empty[-1]['content']
		input_ids_conf4 = tokenizer.apply_chat_template(one_shot_messages_template_empty,add_generation_prompt=True,return_tensors="pt").to(model.device)
	
	
		configurations = [[input_ids_conf1,"conf1",system_conf1,message_conf1]
				  ,[input_ids_conf2,"conf2",system_conf2,message_conf2]
				  ,[input_ids_conf3,"conf3",system_conf3,message_conf3]
				  ,[input_ids_conf4,"conf4",system_conf4,message_conf4]
				 ]

	
		print(configurations[0][2])
	
		for input_ids in configurations:
			writerAnswers = open(AnswersDir+"/"+str(id)+input_ids[1]+"oneShot_answer.txt","a")
			writerQuestions = open(QuestionsDir+"/"+str(id)+input_ids[1]+"oneShot_question.txt","a")
			for i in range(0,5):

				tracker=EmissionsTracker(measure_power_secs=0.1,output_file=ExperimentsDir+"/"+str(id)+input_ids[1]+"oneShot.csv")
				tracker.start()
				tokenized_chat = torch.tensor(input_ids[0][0])
				tokenized_chat = tokenized_chat.unsqueeze(0)
				tokenized_chat = tokenized_chat.to(model.device)
				generated_text = model.generate(tokenized_chat, max_new_tokens=512)

				generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
				tracker.stop()
				
				writerQuestions.write(">>>Start LLM System<<<\n")
				for elem in input_ids[2]:
					writerQuestions.write('role:'+elem['role']+"\n")
					writerQuestions.write('content:'+elem['content']+"\n")
				writerQuestions.write(">>>End LLM System<<<\n")
				writerQuestions.write(">>>Start LLM Prompt<<<\n")
				writerQuestions.write(input_ids[3]+"\n")
				writerQuestions.write(">>>End LLM Prompt<<<\n")
				
				if "assistant" in generated_text:
					generated_text = generated_text.split("assistant")[-1].strip()
					writerAnswers.write(">>>Start LLM Answer<<<\n")
					writerAnswers.write(generated_text+"\n")
					writerAnswers.write(">>>End LLM Answer<<<\n")
				else:
					writerAnswers.write(">>>Start LLM Answer<<<\n")
					writerAnswers.write(generated_text+"\n")
					writerAnswers.write(">>>End LLM Answer<<<\n")
				time.sleep(5)
	
def createFolders(Answers,Questions,Measurements):
	if not os.path.exists(Answers): 
		os.makedirs(Answers)
	if not os.path.exists(Questions): 
		os.makedirs(Questions) 
	if not os.path.exists(Measurements): 
		os.makedirs(Measurements) 


limittexts = 100
baseFolder = "Results2/"
questionsFolder = baseFolder+"AnswersOneShot"
answersFolder = baseFolder+"QuestionsOneShot"
measurementsFolder = baseFolder+"ExperimentsOneShot"
createFolders(questionsFolder,answersFolder,measurementsFolder)
textsDone = recoverState(measurementsFolder)
texts, ids = loadConfigurations(measurementsFolder,AnswersDirs,textsDone)
model,tokenizer = loadModel()
oneShot(texts,ids,model,tokenizer,questionsFolder,answersFolder, measurementsFolder, limittexts)


print('Emissions tracking llama completed')



