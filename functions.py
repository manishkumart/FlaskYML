import PyPDF2
import pandas as pd
import os
import openai
import requests
import re
import time
import io


# openai.api_key = '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'

def generate_nlu_files_from_pdf(pdf_content, api_key):
    
    
    openai.api_key = api_key

    def extract_text_from_pdf(pdf_content):
        text = ""
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        for page in reader.pages:
            text += page.extract_text()
        return text

    extracted_text = extract_text_from_pdf(pdf_content)

    def get_completion(text_content):
        
        user_message = """
        Give me 10 questions that can be answered with the information in text_content.
        What are the corresponding answers?
        """
        
        messages = [
            {'role':'system', 'content': text_content},
            {'role':'user', 'content': f'####{user_message}####'}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=500
        )

        return response.choices[0].message["content"] 

    qa_pairs = get_completion(extracted_text)
    sentences = qa_pairs.split('\n')

    # List of default GPT-3 responses to filter out
    FILTER_RESPONSES = [
    "I can provide five questions and their corresponding answers based on the above text:",
    "I'm sorry, as an AI language model, I cannot provide a list of 10 questions and their corresponding answers without any context. Could you please provide more information or specify the questions you would like me to answer?"
    ]

    # Filter sentences
    sentences = [s for s in sentences if s not in FILTER_RESPONSES and s.strip()]


    df=pd.DataFrame(columns=['question','answer'])

    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            df.loc[len(df)] = {
                'question': sentences[i],
                'answer': sentences[i + 1],
            }
        else:
            df.loc[len(df)] = {
                'question': sentences[i],
                'answer': '',
            }


    df['answer']=df['answer'].apply(lambda x: re.sub(r'Answer: ','',x))
    df['question']=df['question'].apply(lambda x: re.sub(r'\d+\.\s','',x))
    df['answer']=df['answer'].apply(lambda x: re.sub(r'^\d+\.\s*','',x))
    df['question']=df['question'].apply(lambda x: re.sub(r'Q:','',x))
    df['answer']=df['answer'].apply(lambda x: re.sub(r'Answer:','',x))

    # Identify duplicate values in the 'question' column
    duplicate_mask = df.duplicated(subset=['question'], keep='first')

    # Keep only the unique values (first occurrence of each question)
    df_unique = df[~duplicate_mask]

    # Function to get the response from ChatGPT through API key
    def get_completion_from_messages(messages,
                                    model="gpt-3.5-turbo",
                                    temperature=0,
                                    max_tokens=1000):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message["content"]

    # Function to extract entities from question
    def ent_extract(i):
        """
        This function takes question statement from the dataframe of question and answers and return entities
        """
        messages =  [
            {'role':'system',
            'content':"""You are a helpful assistant who performs entity extraction. Your answer should be the primary and  one secondary entity , with nothing else.
            Primary entity should be only program title - or facaulty name like """},
            {'role':'user',
            'content':"""Extract primary and secondary entities from this question statement - {}""".format(i)},
        ]
        response = get_completion_from_messages(messages)
        return response


    entities = [ent_extract(i) for i in df_unique[:]['question']]
    
    df_unique['entities'] = entities

    primary_entities=[]
    secondary_entities=[]

    for i in range(len(df_unique)):
        samples=df_unique['entities'][i].split('Secondary')
        primary_entities.append(samples[0])
        if len(samples)>1:
            secondary_entities.append(samples[1])
        else:
            secondary_entities.append('')


    def preprocessin(entities):
        entities=list(map(lambda x: re.sub('(Primary|Secondary)','',x).strip(),  entities))
        entities=list(map(lambda x: re.sub('Entity','',x),  entities))
        entities=list(map(lambda x: re.sub(r'\s+at Yeshiva University','',x),  entities))
        entities=list(map(lambda x: re.sub(r'\s+at the katz school','',x),  entities))
        entities=list(map(lambda x: re.sub(r'(\s+program\s*|\s+in\s+)',' ',x),  entities))
        entities=list(map(lambda x: re.sub(r'(\s+and\s+)',' ',x),  entities))
        entities=list(map(lambda x: re.sub('(entity|entities)','',x),  entities))
        entities=list(map(lambda x: re.sub('(:|\')','',x),  entities))
        entities = list(map(lambda x: re.sub(r'(\\n|\\n=)', '', x, flags=re.MULTILINE), entities))
        entities = list(map(lambda x: re.sub(r'(masters degree|masters)', 'm.s.', x, flags=re.MULTILINE), entities))
        entities = list(map(lambda x: re.sub(r'(/|\.)', '', x, flags=re.MULTILINE), entities))
        entities = list(map(lambda x: re.sub(r'(/|\.)', '', x, flags=re.MULTILINE), entities))
        entities = list(map(lambda x: x.strip().lower(), entities))

        return entities


    primary_entities = preprocessin(primary_entities)
    secondary_entities = preprocessin(secondary_entities)

    df_unique['primary_entities']=primary_entities
    df_unique['secondary_entities']=secondary_entities
    del df_unique['entities']

    df_unique['unique_intents']=df_unique['primary_entities']+' '+df_unique['secondary_entities']
    df_unique['unique_intents']=df_unique['unique_intents'].apply(lambda x: re.sub(' ','',x))

    def get_completion(prompt, model="gpt-3.5-turbo", retries=3):
        for _ in range(retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=0, # this is the degree of randomness of the model's output
                )
                return response.choices[0].message["content"]
            except openai.error.ServiceUnavailableError as e:
                print(f"Service Unavailable. Retrying after 5 seconds... Error: {e}")
                time.sleep(5)
            except openai.error.APIError as e:
                print(f"API Error. Retrying after 5 seconds... Error: {e}")
                time.sleep(5)
        raise Exception("API call failed after multiple retries.")


    # Function for generating intents with examples for each question

    def intents_examples(question):
        
        text = question
        
        prompt = f"""
        Generate atleast ten similar short questions for the question given in the text

        \"\"\"{text}\"\"\"
        """

        response = get_completion(prompt)
        return response


    # Add a new column 'examples' to store the generated examples
    df_unique['examples'] = ""

    # Iterate over each row in the DataFrame and generate examples for each question
    for idx, row in df_unique.iterrows():
        question = row['question']
        examples = intents_examples(question)
        df_unique.at[idx, 'examples'] = examples


    # Function to generate nlu.yml file for each row

    def generate_nlu_section(df):
        nlu_content = ""
        for _, row in df.iterrows():
            question = row['question']
            unique_intent = row['unique_intents']
            examples = row['examples'].split('\n')
            
            nlu_content += f"- intent: {unique_intent}\n"
            nlu_content += "  examples: |\n"
            
            nlu_content += f"    - {question}\n"
            
            for example in examples:
                if example.strip():
                    # Remove numbering (e.g., "1.", "2.", etc.) from the examples
                    example_text = " ".join(example.split()[1:])
                    nlu_content += f"    - {example_text.strip()}\n"
            
            nlu_content += "\n"
        
        return nlu_content

    # Generate nlu.yml content
    nlu_content = generate_nlu_section(df_unique)

    # Save the nlu content to nlu.yml file
    with open("nlu.yml", "w") as file:
        file.write(nlu_content)


    # Function to generate stories.yml file for each row

    def generate_stories_yaml_row(unique_intents):
        yml_content = (
            f"- story: {unique_intents}\n"
            f"  steps:\n"
            f"  - intent: {unique_intents}\n"
            f"  - action: utter_{unique_intents}\n"
        )
        return yml_content

    # Convert unique intents to YAML format
    yml_content = "\n".join(df_unique.apply(lambda row: generate_stories_yaml_row(row["unique_intents"]), axis=1))

    # Save the YAML content to a file
    with open("stories.yml", "w") as file:
        file.write(yml_content)



    # Functions to generate domain.yml file for each row

    # Function to generate intents section in YAML format
    def generate_intents_section(df):
        #intents = "\n  - ".join(df['unique_intents'])
        intents = "\n- ".join(df['unique_intents'])
        return f"intents:\n- {intents}\n\n"

    # Function to generate actions section in YAML format
    def generate_actions_section(df):
        actions = "\n- ".join([f"utter_{intent}" for intent in df['unique_intents']])
        return f"actions:\n- {actions}\n\n"

    # Function to generate responses section in YAML format
    def generate_responses_section(df):
        responses = "responses:\n"
        for _, row in df.iterrows():
            responses += f"  utter_{row['unique_intents']}:\n  - text: {row['answer']}\n\n"
        return responses

    # Generate YAML content
    yml_content = generate_intents_section(df_unique)
    yml_content += generate_actions_section(df_unique)
    yml_content += generate_responses_section(df_unique)
    

    # Save the YAML content to a file
    with open("domain.yml", "w") as file:
        file.write(yml_content)


# # Example usage
# pdf_path = '/Users/manishkumarthota/projects/katz_flaskapi/manish_k.pdf'  # Replace with the actual path
# api_key = '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' # Replace with your OpenAI API key
# generate_nlu_files_from_pdf(pdf_path, api_key)