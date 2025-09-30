import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer,pipeline
from codecarbon import EmissionsTracker
import time
import json
import os
import marko
import csv
import copy
from bs4 import BeautifulSoup

csv.field_size_limit(1000000)


def loadModel():

	TOKEN=""
	model_id = "meta-llama/CodeLlama-7b-Instruct-hf"


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
	textsDone = []
	for file in os.listdir(ExperimentsDir):
		id = file.split("conf")[0]
		textsDone.append(id)
	return textsDone

def fewShots(texts,ids,model,tokenizer,AnswersDir,QuestionsDir,ExperimentsDir,limittexts):
	
	user_sample1 = '''## Introduction

Rucene is a Rust port of the popular Apache Lucene project. Rucene is not a complete application, but rather a code library and API that can easily be used to add full text search capabilities to applications.

## Status

The index searcher part of Rucene has been put into production and has served all search traffics at Zhihu since July, 2018. Development of the index writer part was started in late 2018, and has been put into production to serve real-time searching since May, 2019.

## Documentation

We don't yet have an API documentation for Rucene, but the usage is similar to [Lucene 6.2.1](https://lucene.apache.org/core/6_2_1/).

> **Note:**
>
> We are working on this, but could use more help since it is a massive project.
'''
	answer_sample1 = 'Rust port of Lucene,"Rucene" - Rust implementation of Lucene'
	
	user_sample2 = '''Log analyser and visualiser for the HotSpot JIT compiler.

* Video introduction to JITWatch [video](https://skillsmatter.com/skillscasts/5243-chris-newland-hotspot-profiling-with-jit-watch)
* Slides from my LJC lightning talk on JITWatch  [slides](http://www.chrisnewland.com/images/jitwatch/HotSpot_Profiling_Using_JITWatch.pdf)

<h3>For instructions and screenshots see the wiki</h3>
<h3>https://github.com/AdoptOpenJDK/jitwatch/wiki</h3>

The JITWatch user interface is built using JavaFX which is downloaded as a maven dependency for JDK11+.

For pre-JDK11 you will need to use a Java runtime that includes JavaFX.

<h2>maven</h2>
<pre>mvn clean compile test exec:java</pre>

<h2>gradle</h2>
<pre>gradlew clean build run</pre>

<h2>Build an example HotSpot log</h2>
<pre># Build the code and then run
./makeDemoLogFile.sh</pre>
'''
	answer_sample2 = 'Log analyser / visualiser for Java HotSpot JIT compiler. Inspect inlining decisions, hot methods, bytecode, and assembly. View results in the JavaFX user interface.","JITWatch"'	

	user_sample3 = ''' 


### **Introduction**
Rabet extension allows you to interact with Stellar Apps in a secure and transparent environment. This product is available for Chrome, Firefox, Edge, and Brave.


### **For users**

Users can easily do all operations available on Stellar, including asset transfer or asset exchange, and interact with all the Stellar apps in a safe user-friendly environment. Rabet is your key to entering the world of Stellar.



### **For developers**

Developers can directly inject Rabet into their application through the browser without the need to install an SDK or package. This is the start of safe interaction with your users.



 
 ### Links
 
Website: https://rabet.io/

Twitter: https://twitter.com/rabetofficial

Discord: https://discord.gg/VkYdnRKUtZ
'''
	answer_sample3 = ':rabbit: :earth_americas: Rabet browser extension enables you to manage your assets and interact with Stellar apps.,"![](https://i.ibb.co/NN1yP1Z/Rabet-banner-2.png)'		

	user_sample4 = '''## Introduction

Rucene is a Rust port of the popular Apache Lucene project. Rucene is not a complete application, but rather a code library and API that can easily be used to add full text search capabilities to applications.

## Status

The index searcher part of Rucene has been put into production and has served all search traffics at Zhihu since July, 2018. Development of the index writer part was started in late 2018, and has been put into production to serve real-time searching since May, 2019.

## Documentation

We don't yet have an API documentation for Rucene, but the usage is similar to [Lucene 6.2.1](https://lucene.apache.org/core/6_2_1/).

> **Note:**
>
> We are working on this, but could use more help since it is a massive project.
'''
	answer_sample4 = 'Rust port of Lucene,"Rucene" - Rust implementation of Lucene'	

	user_sample5 = '''## Introduction

Rucene is a Rust port of the popular Apache Lucene project. Rucene is not a complete application, but rather a code library and API that can easily be used to add full text search capabilities to applications.

## Status

The index searcher part of Rucene has been put into production and has served all search traffics at Zhihu since July, 2018. Development of the index writer part was started in late 2018, and has been put into production to serve real-time searching since May, 2019.

## Documentation

We don't yet have an API documentation for Rucene, but the usage is similar to [Lucene 6.2.1](https://lucene.apache.org/core/6_2_1/).

> **Note:**
>
> We are working on this, but could use more help since it is a massive project.
'''
	answer_sample5 = 'Rust port of Lucene,"Rucene" - Rust implementation of Lucene'	
	
	few_shots_messages_template_base = [
    {
        "role": "system",
        "content": "You are an AI assistant specialized in Github readme summarization. Your task is to summarize the provided readme into a short description. Give only the short description.",
    },
		
	{
		'role': 'user', 
		'content': ''	
	},
		
	{
		'role': 'assistant', 
		'content': ''
	},
		
	{	'role': 'user',
	 	'content': ''
	},
		
	{	'role': 'assistant', 
	 	'content': ''
	},
			
	{	'role': 'user', 
	 	'content': ''
	},
	
	{
		'role': 'assistant', 
		'content': ''
	},
    {
        "role": "user",
        "content": None
    },
	]
	
		
	Request_One_Line = "Hi, summarize the provided Github readme into a short description: "
	
	configurations = []
	
	#First Sample
	message_html_text1 = marko.convert(user_sample1)
	soup1 = BeautifulSoup(message_html_text1)
	message_plain_text1 = soup1.get_text()
	
	#Second Sample
	message_html_text2 = marko.convert(user_sample2)
	soup2 = BeautifulSoup(message_html_text2)
	message_plain_text2 = soup2.get_text()
	
	#Third Sample
	message_html_text = marko.convert(user_sample3)
	soup3 = BeautifulSoup(message_html_text)
	message_plain_text3 = soup3.get_text()
	
	texts = texts[:limittexts]
	
	for text, id in zip(texts,ids):	
		
		html_readme_prompt = marko.convert(text) 
		soup = BeautifulSoup(html_readme_prompt)
		plain_text_prompt = soup.get_text()

		#CONF1 plain
		
		few_shots_messages_conf1 = copy.deepcopy(few_shots_messages_template_base)
		
		#first sample
		few_shots_messages_conf1[1]['content'] = message_plain_text1 # system sample message
		few_shots_messages_conf1[2]['content'] = answer_sample1
		#second sample
		few_shots_messages_conf1[3]['content'] = message_plain_text2 # system sample message
		few_shots_messages_conf1[4]['content'] = answer_sample2
		#third sample
		few_shots_messages_conf1[5]['content'] = message_plain_text3 # system sample message
		few_shots_messages_conf1[6]['content'] = answer_sample3
		
		few_shots_messages_conf1[-1]['content'] = plain_text_prompt # actual testing prompt
		system_conf1 = few_shots_messages_conf1[0:len(few_shots_messages_conf1)-1]
		message_conf1 = few_shots_messages_conf1[-1]['content']
		input_ids_conf1 = tokenizer.apply_chat_template(few_shots_messages_conf1,add_generation_prompt=True,return_tensors="pt").to(model.device)
		
		#CONF2 markdown
		
		few_shots_messages_conf2 = copy.deepcopy(few_shots_messages_template_base)
		
		#first sample
		few_shots_messages_conf2[1]['content'] = user_sample1 # system sample message
		few_shots_messages_conf2[2]['content'] = answer_sample1
		#second sample
		few_shots_messages_conf2[3]['content'] = user_sample2 # system sample message
		few_shots_messages_conf2[4]['content'] = answer_sample2
		#third sample
		few_shots_messages_conf2[5]['content'] = message_plain_text3 # system sample message
		few_shots_messages_conf2[6]['content'] = answer_sample3
		
		few_shots_messages_conf2[-1]['content'] = text # actual testing prompt
		system_conf2 = few_shots_messages_conf2[0:len(few_shots_messages_conf2)-1]
		message_conf2 = few_shots_messages_conf2[-1]['content']
		input_ids_conf2 = tokenizer.apply_chat_template(few_shots_messages_conf2,add_generation_prompt=True,return_tensors="pt").to(model.device)
		
		#CONF3 html
		
		few_shots_messages_conf3 = copy.deepcopy(few_shots_messages_template_base)
		
		#first sample
		few_shots_messages_conf3[1]['content'] = message_html_text1 # system sample message
		few_shots_messages_conf3[2]['content'] = answer_sample1
		#second sample
		few_shots_messages_conf3[3]['content'] = message_html_text2 # system sample message
		few_shots_messages_conf3[4]['content'] = answer_sample2		
		#third sample
		few_shots_messages_conf3[5]['content'] = message_plain_text3 # system sample message
		few_shots_messages_conf3[6]['content'] = answer_sample3
		
		few_shots_messages_conf3[-1]['content'] = html_readme_prompt # actual testing prompt
		system_conf3 = few_shots_messages_conf3[0:len(few_shots_messages_conf3)-1]
		message_conf3 = few_shots_messages_conf3[-1]['content']
		input_ids_conf3 = tokenizer.apply_chat_template(few_shots_messages_conf3,add_generation_prompt=True,return_tensors="pt").to(model.device)
		
		#CONF4 no system
		
		few_shots_messages_conf4 = copy.deepcopy(few_shots_messages_template_base)
		few_shots_messages_conf4[0]['content'] = '' # set system role to empty
		
		#first sample
		few_shots_messages_conf4[1]['content'] = message_plain_text1 # system sample message
		few_shots_messages_conf4[2]['content'] = answer_sample1
		#second sample
		few_shots_messages_conf4[3]['content'] = message_plain_text2 # system sample message
		few_shots_messages_conf4[4]['content'] = answer_sample2
		#third sample
		few_shots_messages_conf4[5]['content'] = message_plain_text3 # system sample message
		few_shots_messages_conf4[6]['content'] = answer_sample3
		
		few_shots_messages_conf4[1]['content'] = user_sample1 # system sample message
		few_shots_messages_conf4[-1]['content'] = Request_One_Line + text # actual testing prompt
		system_conf4 = few_shots_messages_conf4[0:len(few_shots_messages_conf4)-1]
		message_conf4 = few_shots_messages_conf4[-1]['content']
		input_ids_conf4 = tokenizer.apply_chat_template(few_shots_messages_conf4,add_generation_prompt=True,return_tensors="pt").to(model.device)
	
	
		configurations = [[input_ids_conf1,"conf1",system_conf1,message_conf1]
				  ,[input_ids_conf2,"conf2",system_conf2,message_conf2]
				  ,[input_ids_conf3,"conf3",system_conf3,message_conf3]
				  ,[input_ids_conf4,"conf4",system_conf4,message_conf4]
				 ]
	
		for input_ids in configurations:
			writerAnswers = open(AnswersDir+"/"+str(id)+input_ids[1]+"fewShots_answer.txt","a")
			writerQuestions = open(QuestionsDir+"/"+str(id)+input_ids[1]+"fewShots_question.txt","a")
			for i in range(0,5):

				try:
					tracker=EmissionsTracker(measure_power_secs=0.1,output_file=ExperimentsDir+"/"+str(id)+input_ids[1]+"fewShots.csv")
					tracker.start()
					tokenized_chat = torch.tensor(input_ids[0][0])
					tokenized_chat = tokenized_chat.unsqueeze(0)
					tokenized_chat = tokenized_chat.to(model.device)
					generated_text = model.generate(tokenized_chat, max_new_tokens=512)

					generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)

					tracker.stop()
				except:
					writerQuestions.write("Exception")
					writerAnswers.write("Exception")
					continue
				
				writerQuestions.write(">>>Start LLM System<<<\n")
				for elem in input_ids[2]:
					writerQuestions.write('role:'+elem['role']+"\n")
					writerQuestions.write('content:'+elem['content']+"\n")
				writerQuestions.write(">>>End LLM System<<<\n")
				writerQuestions.write(">>>Start LLM Prompt<<<\n")
				writerQuestions.write(input_ids[3]+"\n")
				writerQuestions.write(">>>End LLM Prompt<<<\n")
				
				if "[/INST]" in generated_text:
					generated_text = generated_text.split("[/INST]")[-1].strip()
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
baseFolder = "Results/"
questionsFolder = baseFolder+"AnswersFewShots"
answersFolder = baseFolder+"QuestionsFewShots"
measurementsFolder = baseFolder+"ExperimentsFewShots"
createFolders(questionsFolder,answersFolder,measurementsFolder)
textsDone = recoverState(measurementsFolder)
texts, ids = loadConfigurations(measurementsFolder,AnswersDirs,textsDone)
model,tokenizer = loadModel()
fewShots(texts,ids,model,tokenizer,questionsFolder,answersFolder, measurementsFolder, limittexts)


print('Emissions tracking llama completed')



