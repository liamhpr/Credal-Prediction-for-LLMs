import json

from src import config
import pandas as pd
from datasets import Dataset

with open(f'{config.data_dir}/sets/coqa-dev-v1.0.json', 'r') as infile:
    data = json.load(infile)['data']

dataset = {}
dataset['story'] = []
dataset['question'] = []
dataset['answer'] = []
dataset['additional_answers'] = []
dataset['id'] = []

# loop through every sample in the coqa dev set
for sample_id, sample in enumerate(data):
    story = sample['story']
    questions = sample['questions']
    answers = sample['answers']
    additional_answers = sample['additional_answers']
    
    for question_index, question in enumerate(questions):
        """
        for every question in the sample we add the whole story (the input/text the LLM should use), 
        the question-input-text (the question itself), its answer, the 3 additional answers
        """
        dataset['story'].append(story)
        dataset['question'].append(question['input_text'])
        dataset['answer'].append({
            'text': answers[question_index]['input_text'],
            'answer_start': answers[question_index]['span_start'],
        })

        dataset['id'].append(sample['id'] + '_' + str(question_index))
        additional_answers_list = []

        for i in range(3):
            additional_answers_list.append(additional_answers[str(i)][question_index]['input_text'])

        dataset['additional_answers'].append(additional_answers_list)

        story = story + ' Q: ' + question['input_text'] + ' A: ' + answers[question_index]['input_text']
        if not story[-1] == '.':
            story = story + '.'
        all_answers = [answers[question_index]['input_text']] + additional_answers_list


dataset_df = pd.DataFrame.from_dict(dataset)

dataset = Dataset.from_pandas(dataset_df)

dataset.save_to_disk(f'{config.data_dir}/sets/coqa_dataset')
