import os
import Program_Generator.Parse_Generator

# Change Source and Destination Patjs
q_file_path = os.path.join(os.getcwd(),'Data','Synthetic_Data','Synthetic_Train_Questions.json')
new_file_path = os.path.join(os.getcwd(),'Synthetic_Train_Questions_Programs.json')
Program_Generator.Parse_Generator.generate_programs_from_file(q_file_path, new_file_path, program_style = 'string')
print('Program Generation Complete')