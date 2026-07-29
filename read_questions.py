import csv
from pathlib import Path

CSV_FILE = Path(__file__).parent.joinpath('data/questions.csv')

# get the questions from a csv file and return them in a dictionary
def load_questions(csv_path: Path = CSV_FILE):
    questions = []
    with csv_path.open(newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row or 'number' not in row or 'question' not in row:
                continue
            questions.append({'number': row['number'], 'question': row['question']})
    return questions


if __name__ == '__main__':
    for item in load_questions():
        number = item['number']
        question = item['question']
        if number is not None:
            print(f"{number}. {question}")
        else:
            print(question)
