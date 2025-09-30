import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer,pipeline
from codecarbon import EmissionsTracker
import time
import json
import os
import csv
import marko
from bs4 import BeautifulSoup

csv.field_size_limit(1000000)

def loadModel():

	TOKEN=""
	model_id = "meta-llama/CodeLlama-7b-Instruct-hf"
	#model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

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
			#print("id: "+str(row[0])+"len: "+str(len(row[2])))
			if(str(row[0]) not in textsDone):
				#Readmes.append(entry['input'].replace("<s> ","").replace("</s>","").replace(" . ",".").replace(" .",".").replace(" ;",";"))
				Readmes.append(row[2])
				ids.append(row[0])
	print(len(Readmes))
	return Readmes, ids

		
def recoverState(ExperimentsDir):
	readmesDone = []
	for file in os.listdir(ExperimentsDir):
		id = file.split("conf")[0]
		readmesDone.append(id)
	return readmesDone

def zeroShot(texts,ids,model,tokenizer,AnswersDir,QuestionsDir,ExperimentsDir,limittexts):
	
	zero_shot_messages_template = [
    {
        "role": "system",
        "content": "You are an AI assistant specialized in Github readme summarization. Your task is to summarize the provided readme into a short description. Give only the short description.",
    },
    {
        "role": "user",
        "content": None
    }
	]
	
		
	Request_One_Line = "Hi, summarize the provided Github readme into a short description: "
	configurations = []
	
	texts = texts[:limittexts]
	for text, id in zip(texts,ids):
		
		#last_position = text.rfind(";")
		#reduced_text = text[:last_position+1]
		#uncompleted_line = text[last_position+1:]
		
		html_readme_prompt = marko.convert(text) 
		soup = BeautifulSoup(html_readme_prompt)
		plain_text_prompt = soup.get_text()
		
		#CONF1 plain
		zero_shot_messages_template[-1]['content'] = plain_text_prompt
		system_conf1 = zero_shot_messages_template[0]['content']
		message_conf1 = zero_shot_messages_template[-1]['content']
		
		input_ids_conf1 = tokenizer.apply_chat_template(zero_shot_messages_template,add_generation_prompt=True,return_tensors="pt").to(model.device)
		
		#CONF2 markdown
		zero_shot_messages_template[-1]['content'] = text
		system_conf2 = zero_shot_messages_template[0]['content']
		message_conf2 = zero_shot_messages_template[-1]['content']
		
		input_ids_conf2 = tokenizer.apply_chat_template(zero_shot_messages_template,add_generation_prompt=True,return_tensors="pt").to(model.device)	
		
		#CONF3 html
		zero_shot_messages_template[-1]['content'] = html_readme_prompt
		system_conf3 = zero_shot_messages_template[0]['content']
		message_conf3 = zero_shot_messages_template[-1]['content']
		
		input_ids_conf3 = tokenizer.apply_chat_template(zero_shot_messages_template,add_generation_prompt=True,return_tensors="pt").to(model.device)
		
		#CONF4 no system prompt
		zero_shot_messages_template[-1]['content'] =  Request_One_Line + text
		system_conf4 = ""
		message_conf4 = zero_shot_messages_template[-1]['content']
		
		input_ids_conf4 = tokenizer.apply_chat_template(zero_shot_messages_template,add_generation_prompt=True,return_tensors="pt").to(model.device)	
		
		
	
		configurations = [[input_ids_conf1,"conf1",system_conf1,message_conf1]
						  ,[input_ids_conf2,"conf2",system_conf2,message_conf2]
						  ,[input_ids_conf3,"conf3",system_conf3,message_conf3]
						  ,[input_ids_conf4,"conf4",system_conf4,message_conf4]
						  #,[input_ids_conf5,"conf5",system_conf5,message_conf5]
						 ]

	
		for input_ids in configurations:
			writerAnswers = open(AnswersDir+"/"+str(id)+input_ids[1]+"zeroShot_answer.txt","a")
			writerQuestions = open(QuestionsDir+"/"+str(id)+input_ids[1]+"zeroShot_question.txt","a")
			for i in range(0,5):

				tracker=EmissionsTracker(measure_power_secs=0.1,output_file=ExperimentsDir+"/"+str(id)+input_ids[1]+"zeroShot.csv")
				tracker.start()
				
				tokenized_chat = torch.tensor(input_ids[0][0])
				tokenized_chat = tokenized_chat.unsqueeze(0)
				tokenized_chat = tokenized_chat.to(model.device)
				generated_text = model.generate(tokenized_chat, max_new_tokens=512)

				generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)

				tracker.stop()
				
				writerQuestions.write(">>>Start LLM System<<<\n")
				writerQuestions.write(input_ids[2]+"\n")
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
questionsFolder = baseFolder+"AnswersZeroShot"
answersFolder = baseFolder+"QuestionsZeroShot"
measurementsFolder = baseFolder+"ExperimentsZeroShot"
createFolders(questionsFolder,answersFolder,measurementsFolder)
textsDone = recoverState(measurementsFolder)
texts, ids = loadConfigurations(measurementsFolder,AnswersDirs,textsDone)
model,tokenizer = loadModel()
zeroShot(texts,ids,model,tokenizer,questionsFolder,answersFolder, measurementsFolder, limittexts)

print('Emissions tracking llama completed')



