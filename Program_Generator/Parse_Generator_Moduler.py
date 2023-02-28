import os
import json

def load_data(file_path):
    """Loads JSON file for Questions
       Input:
            file_path (str): Path to Question JSON file
       Return:
            questions (list): List of All Questions
            question_types (list): List of All Question Types
            programs (list): List of All Programs
            unique_questions (list): List of Unique Questions across Question File
            unique_question_types (list): List of Unique Question types across Question File
            unique_programs (list): List of Unique Programs across Question File
    """
    file = open(os.path.join(file_path))
    data = json.load(file)

    questions = []
    question_types = []
    programs = []
    ground_truths = []
    unique_questions = []
    unique_question_types = []
    unique_programs = []
    unique_ground_truths = []

    for key in list(data.keys()):
        q = data[key]['Question']
        q_type = data[key]['Question_Type']
        program = data[key]['Program']
        gt = data[key]['Ground_Truth']
        questions.append(q)
        question_types.append(q_type)
        programs.append(program)
        ground_truths.append(gt)
        if q not in unique_questions:
            unique_questions.append(q)
        if q_type not in unique_question_types:
            unique_question_types.append(q_type)
        if program not in unique_programs:
            unique_programs.append(program)
        if gt not in unique_ground_truths:
            unique_ground_truths.append(gt)

    return questions, question_types, programs, ground_truths, unique_questions, unique_question_types, unique_programs, unique_ground_truths

def program_generator(question, output_type = 'list'):
     """This function takes a question string and returns a sequence of functions 
          (i.e program string to be executed by the execution engine)
          Input:
               question (str): Question String
               output_type (str): Options for return type ('list','string') [If 'string' then use separator ' & ']
          Output:
               program (list): A list (sequence) of functions 
     """
     function_list = {'image_classify':'Classify_Overall_Flood_Condition',
                      'count':'Count',
                      'exists':'Exists',
                      'filter_buildings':'Filter(Buildings)',
                      'filter_road':'Filter(Road)',
                      'filter_flooded':'Filter(Flooded)',
                      'filter_non-flooded':'Filter(Non_Flooded)'}

     # Keywords
     starting_word_list = ['What', 'How', 'Is']
     attribute_list = ['flooded', 'non-flooded']
     object_list = ['buildings', 'road']
     special_word_list = ['condition','overall']

     # Preprocessing Question
     question = question.replace("?","")
     question = question.replace("non flooded","non-flooded")
     q_tokens = question.split(" ")

     # Parse Question
     start = q_tokens[0]
     attribute = None
     object = None
     special_word = []
     for attr in attribute_list:
          if attr in q_tokens:
              attribute = attr
              break
     for obj in object_list:
          if obj in q_tokens:
              object = obj
              break
     for spcl_word in special_word_list:
          if spcl_word in q_tokens:
              special_word.append(spcl_word)

     # Generate Program
     program_list = []
     program_string = ''
     if 'overall' in special_word:
          program_list.append(function_list['image_classify'])
          program_string += function_list['image_classify']
          if output_type == 'list':
               return program_list
          elif output_type == 'string':
               return program_string
     else:
          if attribute != None and object!= None:
               program_list.append(function_list['filter_' + object])
               program_list.append(function_list['filter_' + attribute])
               program_string += function_list['filter_' + object] + ' & ' + function_list['filter_' + attribute]
          elif attribute == None and object!= None:
               program_list.append(function_list['filter_' + object])
               program_string += function_list['filter_' + object]
          
          if start == 'How':
               program_list.append(function_list['count'])
               program_string += ' & '
               program_string += function_list['count']
          elif start == 'Is':
               program_list.append(function_list['exists'])
               program_string += ' & '
               program_string += function_list['exists']
          
          if output_type == 'list':
               return program_list
          elif output_type == 'string':
               return program_string

def generate_programs_from_file(file_path, new_file_path, program_style = 'list'):
     """Generates Programs for all Questions in a given file and 
        returns the file with programs added in the JSON file
        Input:
             file_path (str): Path to Question JSON file
             new_file_path (str): Path to New Question JSON file where Questions and Programs are stored
             program_style (str): Options for return type ('list','string') [If 'string' then use separator ' & ']
        Output:
             Json file with programs to each questions added in the file
     """
     question_file = open(os.path.join(file_path))
     data = json.load(question_file)
     with open(new_file_path,'w') as new_file:
        for key in list(data.keys()):
          q = data[key]['Question']
          data[key]['Program'] = program_generator(q, output_type = program_style)
        json.dump(data, new_file)