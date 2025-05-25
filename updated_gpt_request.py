#!/usr/bin/env python3
import argparse
import fnmatch
import json
import os
import pdb
import pickle
import re
import sqlite3
from typing import Dict, List, Tuple

import backoff
import openai
import pandas as pd
import sqlparse
from tqdm import tqdm
'''openai configure'''

openai.debug=True


def new_directory(path):  
    if not os.path.exists(path):  
        os.makedirs(path)  


def get_db_schemas(bench_root: str, db_name: str) -> Dict[str, str]:
    """
    Read an sqlite file, and return the CREATE commands for each of the tables in the database.
    """
    asdf = 'database' if bench_root == 'spider' else 'databases'
    with sqlite3.connect(f'file:{bench_root}/{asdf}/{db_name}/{db_name}.sqlite?mode=ro', uri=True) as conn:
        # conn.text_factory = bytes
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schemas = {}
        for table in tables:
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
            schemas[table[0]] = cursor.fetchone()[0]

        return schemas

def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]

    # Print the column names
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    # print    combined_prompts = "\n\n".join([schema_prompt, comment_prompt, difficulty_guidance,final_instruction]).strip(o(header)
    # Pri    combined_prompts = "\n\n".join([schema_prompt, comment_prompt, difficulty_guidance,final_instruction]).strip(int the values
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output

def generate_schema_prompt(db_path, num_rows=None):
    # extract create ddls
    '''
    :param root_place:
    :param db_name:
    :return:
    '''
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table == 'sqlite_sequence':
            continue
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        if num_rows:
            cur_table = table[0]
            if cur_table in ['order', 'by', 'group']:
                cur_table = "`{}`".format(cur_table)

            cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(num_rows, cur_table, num_rows, rows_prompt)
            schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt

def generate_comment_prompt(question, knowledge=None):
    pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_prompt_kg = "-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above."
    # question_prompt = "-- {}".format(question) + '\n SELECT '
    question_prompt = "-- {}".format(question)
    knowledge_prompt = "-- External Knowledge: {}".format(knowledge)

    if not knowledge_prompt:
        result_prompt = pattern_prompt_no_kg + '\n' + question_prompt
    else:
        result_prompt = knowledge_prompt + '\n' + pattern_prompt_kg + '\n' + question_prompt

    return result_prompt

def cot_wizard():
    cot = "\nGenerate the SQL after thinking step by step: "
    
    return cot

def few_shot():
    ini_table = "CREATE TABLE singer\n(\n    singer_id         TEXT not null\n        primary key,\n    nation       TEXT  not null,\n    sname       TEXT null,\n    dname       TEXT null,\n    cname       TEXT null,\n    age    INTEGER         not null,\n    year  INTEGER          not null,\n    birth_year  INTEGER          null,\n    salary  REAL          null,\n    city TEXT          null,\n    phone_number   INTEGER          null,\n--     tax   REAL      null,\n)"
    ini_prompt = "-- External Knowledge: age = year - birth_year;\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- How many singers in USA who is older than 27?\nThe final SQL is: Let's think step by step."
    ini_cot_result = "1. referring to external knowledge, we need to filter singers 'by year' - 'birth_year' > 27; 2. we should find out the singers of step 1 in which nation = 'US', 3. use COUNT() to count how many singers. Finally the SQL is: SELECT COUNT(*) FROM singer WHERE year - birth_year > 27;</s>"
    
    one_shot_demo = ini_table + '\n' + ini_prompt + '\n' + ini_cot_result
    
    return one_shot_demo

def few_shot_no_kg():
    ini_table = "CREATE TABLE singer\n(\n    singer_id         TEXT not null\n        primary key,\n    nation       TEXT  not null,\n    sname       TEXT null,\n    dname       TEXT null,\n    cname       TEXT null,\n    age    INTEGER         not null,\n    year  INTEGER          not null,\n    age  INTEGER          null,\n    salary  REAL          null,\n    city TEXT          null,\n    phone_number   INTEGER          null,\n--     tax   REAL      null,\n)"
    ini_prompt = "-- External Knowledge:\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- How many singers in USA who is older than 27?\nThe final SQL is: Let's think step by step."
    ini_cot_result = "1. 'older than 27' refers to age > 27 in SQL; 2. we should find out the singers of step 1 in which nation = 'US', 3. use COUNT() to count how many singers. Finally the SQL is: SELECT COUNT(*) FROM singer WHERE age > 27;</s>"
    
    one_shot_demo = ini_table + '\n' + ini_prompt + '\n' + ini_cot_result
    
    return one_shot_demo


def generate_schema_prompt_with_discovery(db_path, num_rows=None):
    # Assumes existing function to generate schema prompt
    schema_prompt = generate_schema_prompt(db_path, num_rows=num_rows)
    discovery_guide = "-- Review the schema: Identify key tables and relationships. What data might you need?"
    return "\n".join([schema_prompt, discovery_guide]).strip()

def detailed_self_discovery_guide():
    steps = ["-- Step 1: Analyze the schema and identify all tables and columns relevant to the question.",
             "-- Step 2: Determine how these tables are related. Identify the type of joins needed to connect these tables.",
             "-- Step 3: Outline what information you need to extract from each table. Consider what columns need to be selected or calculated.",
             "-- Step 4: Identify any filters or conditions that need to be applied to the data. This includes WHERE clauses and any necessary data transformations.",
             "-- Step 5: Consider if any data needs to be aggregated. Determine what GROUP BY or HAVING clauses might be necessary.",
             "-- Step 6: Plan how the results should be ordered. Decide on the ORDER BY clauses.",
             "-- Step 7: Think about the final presentation of the data. Do you need a subquery? Should you use WITH clauses for readability?",
             "-- Step 8: Construct the SQL query step by step, incorporating the elements identified in the previous steps."                                                                             ]
    return '\n'.join(steps)


'''def generate_difficulty_based_prompt(db_path, question, knowledge=None):
    """
    Generates a prompt based on the difficulty level of the question, incorporating schema details,
    external knowledge, and specific guidance for handling the question based on its difficulty.
    """                                                            
    schema_prompt = generate_schema_prompt_with_discovery(db_path)
    comment_prompt = generate_comment_prompt(question, knowledge)
    final_instruction = "Given the database schema and any additional information provided above, please write the SQL query that answers the following question."
    difficulty_guidance = detailed_self_discovery_guide()
    question_prompt = f"Question: {question}"
    #combined_prompt = "\n\n".join([schema_prompt, comment_prompt,question_prompt, difficulty_guidance, final_instruction])
    combined_prompt = f"{schema_prompt}\n\n{comment_prompt}\n\n{difficulty_guidance}\n\n{final_instruction}\n\n{question_prompt}\n\n-- Your SQL query starts below this line:\nSELECT"
    return combined_prompt'''




# Use the below function to use  attempt history for feeding the history to the model and generating prompt

def generate_difficulty_based_prompt(db_path, question, knowledge=None, difficulty='simple', attempts_history=[]):
    """
    Generates a prompt based on the difficulty level of the question, incorporating schema details,
    external knowledge, and specific guidance for handling the question based on its difficulty.
    """
    schema_prompt = generate_schema_prompt_with_discovery(db_path)
    comment_prompt = generate_comment_prompt(question, knowledge)
    #difficulty_guidance = detailed_self_discovery_guide()
    final_instruction = "Given the database schema and any additional information provided above, please write the SQL query that answers the following question."
    history_summary = "\n\nPrevious Attempts Summary are given below. Please look at the scores and the generated SQL. Try to do better than this:\n"
    
    if attempts_history and len(attempts_history) > 0:
        for idx, attempt in enumerate(attempts_history, start=1):
            attempt_sql = attempt['generated_sql'].split('\t----- bird -----')[0]  # Exclude metadata for readability
            history_summary += f"Attempt {idx}: Similarity {attempt['similarity_percentage']}%, Semantically Correct: {'Yes' if attempt['is_semantically_correct'] else 'No'}\nSQL: {attempt_sql}\n\n"
    else:
        history_summary += "No previous attempts."
    
    if difficulty == 'simple':
        difficulty_guidance = detailed_self_discovery_guide()#"-- Given the simplicity of this question, focus on selecting the right columns and applying basic conditions directly."
    elif difficulty == 'moderate':
        difficulty_guidance = detailed_self_discovery_guide()#"-- This question is of moderate difficulty. Consider the relationships between tables, and you may need to use joins. Think about what intermediate steps are required to connect the data."
    elif difficulty == 'challenging':
        difficulty_guidance =  detailed_self_discovery_guide()
    #else:
       # difficulty_guidance = detailed_self_discovery_guide()
    question_prompt = f"Question: {question}"                                                  
    #combined_prompt = "\n\n".join([schema_prompt, comment_prompt,question_prompt, difficulty_guidance, final_instruction])
    combined_prompt = f"{schema_prompt}\n\n{comment_prompt}\n\n{difficulty_guidance}\n\n{history_summary}\n\n{final_instruction}\n\n{question_prompt}\n\n-- Your SQL query starts below this line:\nSELECT"
    return combined_prompt



def generate_combined_prompts_one(db_path, question, knowledge=None):
    schema_prompt = generate_schema_prompt(db_path, num_rows=None) # This is the entry to collect values
    comment_prompt = generate_comment_prompt(question, knowledge)

    combined_prompts = schema_prompt + '\n\n' + comment_prompt + cot_wizard() + '\nSELECT '
    # combined_prompts = few_shot() + '\n\n' + schema_prompt + '\n\n' + comment_prompt

    # print(combined_prompts)

    return combined_prompts

def quota_giveup(e):
    return isinstance(e, openai.error.RateLimitError) and "quota" in str(e)

@backoff.on_exception(
    backoff.constant,
    openai.error.OpenAIError,
    giveup=quota_giveup,
    raise_on_giveup=True,
    interval=20
)

def extract_correct_sql_from_json(eval_path: str) -> List[str]:
    with open(eval_path, 'r') as file:
        data_json = json.load(file)

    correct_sql_list = []
    for data in data_json:
        correct_sql = data.get('SQL')
        if correct_sql is None:
            raise ValueError(f"Missing 'SQL' field for data with question ID {data.get('question_id')}")
        correct_sql_list.append(correct_sql)

    return correct_sql_list

import sqlparse

def normalize_sql(sql):
    """ Normalize SQL query for comparison. """
    sql = sql.lower()
    sql = sqlparse.format(sql, reindent=True, keyword_case='upper')
    return ' '.join(sql.split())

import difflib


import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

def get_sql_components(sql):
    """ Breaks down an SQL query into its constituent components. """
    parsed = sqlparse.parse(sql)[0]
    components = {}
    for token in parsed.tokens:
        if token.ttype is DML:
            components['DML'] = token.value
        if token.ttype is Keyword:
            components[token.value.upper()] = []
        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                components[token.get_real_name()].append(identifier.value)
        elif isinstance(token, Identifier):
            components[token.get_real_name()] = [token.value]
    return components

def calculate_sql_similarity(sql1, sql2):
    """ Calculate similarity based on SQL components. """
    components1 = get_sql_components(sql1)
    components2 = get_sql_components(sql2)

    common_components = set(components1.keys()) & set(components2.keys())
    total_components = set(components1.keys()) | set(components2.keys())

    similarity_count = 0
    for component in common_components:
        if components1[component] == components2[component]:
            similarity_count += 1

    return (similarity_count / len(total_components)) * 100
def calculate_similarity_percentage(str1, str2):
    """ Calculate the similarity percentage between two strings. """
    return difflib.SequenceMatcher(None, str1, str2).ratio() * 100

# For strict semantic comparison

'''def semantic_comparison(generated_sql, correct_sql, db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            # Execute and fetch results from the generated SQL query
            gen_cursor = conn.cursor()
            gen_cursor.execute(generated_sql)
            gen_results = gen_cursor.fetchall()

            # Execute and fetch results from the correct SQL query
            corr_cursor = conn.cursor()
            corr_cursor.execute(correct_sql)
            corr_results = corr_cursor.fetchall()

            return gen_results == corr_results
    except Exception as e:
        print(f"Error during semantic comparison: {e}")
        return False '''


'''def semantic_comparison(generated_sql, correct_sql, db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(generated_sql)
            gen_results =cursor.fetchall()
            cursor.execute(correct_sql)
            corr_results = cursor.fetchall()  # Sort results for comparison

            return gen_results == corr_results
    except Exception as e:
        print(f"Error during semantic comparison for query '{generated_sql}': {e}")
        return False'''

from collections import Counter


def normalize_row(row):
    normalized = []
    for value in row:
        try:
            normalized.append(float(value) if '.' in value else int(value))
        except (ValueError, TypeError):
            normalized.append(value)
    return tuple(normalized)

def semantic_comparison(generated_sql, correct_sql, db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(generated_sql)
            gen_results = [normalize_row(row) for row in cursor.fetchall()]
            cursor.execute(correct_sql)
            corr_results = [normalize_row(row) for row in cursor.fetchall()]
            return Counter(gen_results) == Counter(corr_results)
    except Exception as e:
        print(f"Error during semantic comparison for query '{generated_sql}': {e}")
        return False


    
import openai


'''def generate_prompt_with_history(schema_prompt, question, difficulty, knowledge=None, attempts_history=[]):
    history_summary = "Previous Attempts Summary:\n"
    for attempt in attempts_history:
        attempt_summary = f"- Attempt {attempt['attempt_number']}: Similarity Score: {attempt['similarity_percentage']}%. "
        if attempt['is_semantically_correct']:
            attempt_summary += "This attempt was semantically correct."
        else:
            attempt_summary += "This attempt was not semantically correct. Feedback: " + attempt['feedback']
        history_summary += attempt_summary + "\n"

    knowledge_section = f"-- External Knowledge: {knowledge}\n" if knowledge else ""
    difficulty_section = f"-- Difficulty Level: {difficulty}. Specific Guidance: {detailed_self_discovery_guide()}\n"

    prompt = f"{schema_prompt}\n\n{knowledge_section}{difficulty_section}\n{history_summary}\n-- Given the above schema and feedback, please generate an improved SQL query for: {question}\n\n-- Your SQL query starts below this line:\nSELECT"

    return prompt'''



'''Use the below function for Completion End point Models'''
def connect_gpt(engine, prompt, max_tokens, temperature, stop):
    #print("Prompt to GPT:")
    #print(prompt)
    #print("-" * 50)
    try:
        result = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
    except Exception as e:
        result = 'error:{}'.format(e)
    return result



''' Use the below function for generating feedback on upto three trials, not feeding feedback history to the model'''

'''def collect_response_from_gpt(db_path_list, question_list, api_key, engine, knowledge_list=None, eval_path=None):
    openai.api_key = api_key
    hardcoded_json_path = '/home/shouvon/DAMO-ConvAI/bird/llm/data/dev/dev2.json'
    correct_sql_list = extract_correct_sql_from_json(hardcoded_json_path) if hardcoded_json_path else [None] * len(question_list)
    feedback_results = {}
    response_list = []
    for i, question in enumerate(question_list):
        best_attempt = {"similarit    combined_prompts = "\n\n".join([schema_prompt, comment_prompt, difficulty_guidance,final_instruction]).strip(iy_percentage": 0, "is_semantically_correct": False, "generated_sql": None, "attempt_count": 0}
        attempts = 0
        attempts_history = []
        for attempt_number in range(1, 4):
            prino(f"Processing {i}th question, attempt {attempts+1}: {question}")
            knowledge = knowledge_list[i] if knowledge_list else None
            difficulty = difficulty_list[i] if difficulty_list else 'simple'
            cur_prompt = generate_difficulty_based_prompt(db_path=db_path_list[i], question=question, knowledge=knowledge, difficulty=difficulty)

            result = connect_gpt(engine=engine, prompt=cur_prompt, max_tokens=256, temperature=0.5, stop=['--', '\n\n', ';', '#'])

            if type(result) == str:
                sql = result
            else:
                sql = 'SELECT' + result['choices'][0]['text']

            db_id = db_path_list[i].split('/')[-1].split('.sqlite')[0]
            sql = sql + '\t----- bird -----\t' + db_id # to avoid unpredicted \t appearing in codex results
            #response_list.append(sql)
            if sql:  # Check if SQL is generated
                normalized_sql = normalize_sql(sql)  # Keep the SQL format consistent for semantic comparison
                similarity_percentage = calculate_similarity_percentage(normalized_sql, correct_sql_list[i])

                if similarity_percentage > best_attempt["similarity_percentage"]:
                    is_semantically_correct = semantic_comparison(normalized_sql, correct_sql_list[i], db_path_list[i])
                    if is_semantically_correct or attempts == 2:  # Save if semantically correct or on last attempt
                        best_attempt = {
                                "generated_sql": sql,
                                "similarity_percentage": similarity_percentage,
                                "is_semantically_correct": is_semantically_correct,
                                "attempt_count": attempts + 1
                            }
                        attempts_history.append(best_attempt)
                        if is_semantically_correct:
                            break  # Stop if semantically correct
            attempts += 1

        feedback_results[question] = best_attempt
        if best_attempt["generated_sql"]:
            response_list.append(best_attempt["generated_sql"])

    return response_list, feedback_results'''


'''Use the below function to use Chat Completion End point models'''

'''def connect_gpt(engine, prompt, max_tokens, temperature, stop):
    #print("Prompt to GPT:")
    #print(prompt)
    #print("-" * 50)

    client = openai.ChatCompletion()
    try:
        result = client.create( model=engine, max_tokens=max_tokens, messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}],temperature=temperature, stop=stop)
    except Exception as e:
        result = 'error:{}'.format(e)
    return result'''

'''Use the below function if you need to feed the feedback and generated sql history to the model with self discover'''

def collect_response_from_gpt(db_path_list, question_list, api_key, engine,difficulty_list=None, knowledge_list=None, eval_path=None):
    openai.api_key = api_key
    hardcoded_json_path = '/home/shouvon/DAMO-ConvAI/bird/llm/data/dev/dev.json'
    correct_sql_list = extract_correct_sql_from_json(hardcoded_json_path) if hardcoded_json_path else [None] * len(question_list)


    #correct_sql_list = extract_correct_sql_from_json(eval_path) if eval_path else [None] * len(question_list)
    feedback_results = {}
    response_list = []

    for i, question in enumerate(question_list):
        best_attempt = {"similarity_percentage": 0, "is_semantically_correct": False, "generated_sql": None, "attempt_count": 0}
        attempts_history = []
        
        for attempt_number in range(1, 4):
            print(f"Processing {i}th question, attempt {attempt_number}: {question}")
            knowledge = knowledge_list[i] if knowledge_list else None
            difficulty = difficulty_list[i] if difficulty_list else 'simple'
            
            # Now include attempts_history in the prompt
            cur_prompt = generate_difficulty_based_prompt(db_path=db_path_list[i], question=question, knowledge=knowledge, difficulty=difficulty, attempts_history=attempts_history)
            
            result = connect_gpt(engine=engine, prompt=cur_prompt, max_tokens=256, temperature=0.5, stop=['--', '\n\n', ';', '#'])
            #print(result)
            if type(result) == str:
                sql = result
            else:
                sql = 'SELECT ' + result['choices'][0]['text']

            db_id = db_path_list[i].split('/')[-1].split('.sqlite')[0]
            sql = sql + '\t----- bird -----\t' + db_id # to avoid unpredicted \t appearing in codex results

            #sql = result['choices'][0]['text'] if 'choices' in result else ""
            #full_sql = f"{sql}\t----- bird -----\t{db_id}"

            normalized_sql = normalize_sql(sql)
            similarity_percentage = calculate_similarity_percentage(normalized_sql, correct_sql_list[i])
            is_semantically_correct = semantic_comparison(normalized_sql, correct_sql_list[i], db_path_list[i])

            current_attempt = {
                "generated_sql":sql,
                "similarity_percentage": similarity_percentage,
                "is_semantically_correct": is_semantically_correct,
                "attempt_count": attempt_number
            }

            # Update attempts_history with details of the current attempt
            attempts_history.append(current_attempt)
            
            # Determine if this is the best attempt so far
            if is_semantically_correct or similarity_percentage > best_attempt["similarity_percentage"]:
                best_attempt = current_attempt
                
                # Stop if semantically correct
                if is_semantically_correct:
                    break

        feedback_results[question] = best_attempt
        if best_attempt["generated_sql"]:
            response_list[str(i)] = best_attempt["generated_sql"]

    return response_list, feedback_results 


'''def collect_response_from_gpt(db_path_list, question_list, api_key, engine, knowledge_list=None):
    responses_dict = {}
    response_list = []
    openai.api_key = api_key
    for i, question in tqdm(enumerate(question_list)):
        print('--------------------- processing {}th question ---------------------'.format(i))
        print('the question is: {}'.format(question))
        
        if knowledge_list:
            cur_prompt = generate_difficulty_based_prompt(db_path=db_path_list[i], question=question, knowledge=knowledge_list[i])
        else:
            cur_prompt = generate_difficulty_based_prompt(db_path=db_path_list[i], question=question)
        
        plain_result = connect_gpt(engine=engine, prompt=cur_prompt, max_tokens=256, temperature=0, stop=['--', '\n\n', ';', '#'])
        # pdb.set_trace()
        # plain_result = connect_gpt(engine=engine, prompt=cur_prompt, max_tokens=256, temperature=0, stop=['</s>'])
        # determine wheter the sql is wrong
        
        if type(plain_result) == str:
            sql = plain_result
        else:
            sql = 'SELECT' + plain_result['choices'][0]['text']
        
        # responses_dict[i] = sql
        db_id = db_path_list[i].split('/')[-1].split('.sqlite')[0]
        sql = sql + '\t----- bird -----\t' + db_id # to avoid unpredicted \t appearing in codex results
        response_list.append(sql)

    return response_list'''




def question_package(data_json, knowledge=False):
    question_list = []
    for data in data_json:
        question_list.append(data['question'])

    return question_list

def knowledge_package(data_json, knowledge=False):
    knowledge_list = []
    for data in data_json:
        knowledge_list.append(data['evidence'])

    return knowledge_list

def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    #difficulty_list = [] #Use this when you will need difficulty based adaptive prompting
    for i, data in enumerate(datasets):
        question_list.append(data['question'])
        cur_db_path = db_root_path + data['db_id'] + '/' + data['db_id'] +'.sqlite'
        db_path_list.append(cur_db_path)
        knowledge_list.append(data['evidence'])
        #difficulty_list.append(data['difficulty'])
    
    return question_list, db_path_list, knowledge_list, #difficulty_list

def generate_sql_file(sql_lst, output_path=None):
    result = {}
    for i, sql in enumerate(sql_lst):
        result[i] = sql
    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        json.dump(result, open(output_path, 'w'), indent=4)
    return result


'''def save_feedback(feedback_results, output_path):
    with open(output_path, 'w') as file:
        for question, result in feedback_results.items():
            feedback_str = f"Generated SQL for '{question}' is "
            if result["is_semantically_correct"]:
                feedback_str += "correct semantically and has a similarity percentage of "
            else:
                feedback_str += "incorrect. "
            feedback_str += f"{result['similarity_percentage']}%."
            if result['attempt_count'] == 3 and not result['is_semantically_correct']:
                feedback_str += " Maximum attempts reached."
            file.write(feedback_str + '\n')'''


if __name__ == '__main__':
    # Argument parsing
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--eval_path', type=str, default='')
    args_parser.add_argument('--mode', type=str, default='dev')
    args_parser.add_argument('--test_path', type=str, default='')
    args_parser.add_argument('--use_knowledge', type=str, default='False')
    args_parser.add_argument('--db_root_path', type=str, default='')
    args_parser.add_argument('--api_key', type=str, required=True)
    args_parser.add_argument('--engine', type=str, required=True, default='code-davinci-002')
    args_parser.add_argument('--data_output_path', type=str)
    args_parser.add_argument('--chain_of_thought', type=str)
    args = args_parser.parse_args()
    
    # Load evaluation data
    eval_data = json.load(open(args.eval_path, 'r'))
    
    # Decouple question and schema
    question_list, db_path_list, knowledge_list, = decouple_question_schema(datasets=eval_data, db_root_path=args.db_root_path)
    assert len(question_list) == len(db_path_list) == len(knowledge_list)

    # Collect responses from GPT
    if args.use_knowledge == 'True':
        responses  = collect_response_from_gpt(db_path_list=db_path_list, question_list=question_list, api_key=args.api_key, engine=args.engine, knowledge_list=knowledge_list)
    else:
        responses = collect_response_from_gpt(db_path_list=db_path_list, question_list=question_list, api_key=args.api_key, engine=args.engine, knowledge_list=None)
    
    # Save SQL queries
    output_name = args.data_output_path + 'predict_' + args.mode + ('_cot.json' if args.chain_of_thought == 'True' else '.json')
    generate_sql_file(sql_lst=responses, output_path=output_name)

    # Save feedback
    #feedback_file_path = './feedback_results.txt'
    #save_feedback(feedback_results, feedback_file_path)
    print('Successfully collected results from {} for {} evaluation; Use knowledge: {}; Use COT: {}'.format(args.engine, args.mode, args.use_knowledge, args.chain_of_thought))
