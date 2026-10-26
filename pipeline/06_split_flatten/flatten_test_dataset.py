import os

import json
from pathlib import Path

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

def flatten_test_datasets():
    data_dir = DATA / "06_split"
    datasets = ['cognition', 'reasoning', 'awareness']
    
    for ds in datasets:
        input_file = data_dir / f'saudade_{ds}_test.json'
        if not input_file.exists():
            input_file = data_dir / f'saudade_{ds}_pt_test.json'
            
        output_file = data_dir / f'saudade_{ds}_test_flat.jsonl'
        output_file_baseline = data_dir / f'saudade_{ds}_test_flat_baseline.jsonl'
        
        if not input_file.exists(): continue
            
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        flat_items = []
        flat_items_baseline = []
        
        for category, entities in data.items():
            for entity, content in entities.items():
                if ds == 'awareness':
                    qa_list = content.get('awareness_qa', [])
                    if not qa_list:
                        for _, ev_content in content.items():
                            if isinstance(ev_content, dict) and 'awareness_qa' in ev_content:
                                qa_list.extend(ev_content['awareness_qa'])
                    
                    for qa_item in qa_list:
                        past_fact = qa_item.get('past_fact', '')
                        future_fact = qa_item.get('future_fact', '')
                        future_date = qa_item.get('future_date', '')
                        
                        inner_qa = qa_item.get('future_questions', qa_item.get('future_question', {}))
                        if not inner_qa: inner_qa = qa_item.get('questions', qa_item.get('question', {}))
                        
                        if isinstance(inner_qa, str): inner_qa = {'generic': inner_qa}
                        
                        for qt, qtext in inner_qa.items():
                            if 'generic' not in qt: continue
                            if qtext:
                                question_with_context = qtext
                                if past_fact and future_date:
                                    question_with_context = (
                                        f'Data de hoje: {future_date}\n\n'
                                        f'Dado o seguinte contexto, responde à pergunta no final. '
                                        f'A tua resposta deve conter apenas o nome ou valor, sem mais palavras.\n\n'
                                        f'CONTEXTO: {past_fact}\n\n'
                                        f'PERGUNTA: {qtext}'
                                    )
                                flat_items.append({
                                    'question': question_with_context,
                                    'expected_answers': qa_item.get('future_answers', [qa_item.get('future_ground_truth', '')]),
                                    'distractor_fact': past_fact,
                                    'target_fact': future_fact,
                                    'entity': entity,
                                    'category': category,
                                })
                                flat_items_baseline.append({
                                    'question': qtext,
                                    'expected_answers': qa_item.get('future_answers', [qa_item.get('future_ground_truth', '')]),
                                    'distractor_fact': 'N/A',
                                    'target_fact': future_fact,
                                    'entity': entity,
                                    'category': category,
                                })
                elif ds == 'cognition':
                    cog_content = content.get('cognition_qa', content)
                    for event_title, ev_content in cog_content.items():
                        if not isinstance(ev_content, dict): continue
                        questions = ev_content.get('questions') or ev_content.get('question')
                        answers = ev_content.get('answers', [])
                        if questions:
                            if isinstance(questions, str): questions = {'generic': questions}
                            for qt, qtext in questions.items():
                                if 'generic' not in qt: continue
                                if qtext:
                                    item_payload = {'question': qtext, 'expected_answers': answers, 'entity': entity, 'category': category}
                                    flat_items.append(item_payload)
                                    flat_items_baseline.append(item_payload)
                elif ds == 'reasoning':
                    outer_rqa = content.get('reasoning_qa', content.get('ranking_qa', []))
                    for pair_item in outer_rqa:
                        inner_rqa = pair_item.get('ranking_qa', {})
                        answers = pair_item.get('answers', [])
                        facts_pair = pair_item.get('facts_pair', [])
                        for qt, qtext in inner_rqa.items():
                            if 'generic' not in qt: continue
                            if qtext:
                                item_payload = {'question': qtext, 'expected_answers': answers, 'entity': entity, 'category': category, 'facts_pair': facts_pair}
                                flat_items.append(item_payload)
                                flat_items_baseline.append(item_payload)
                                
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in flat_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        with open(output_file_baseline, 'w', encoding='utf-8') as f:
            for item in flat_items_baseline:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        print(f'Flattened {ds} test set: {len(flat_items)} items -> {output_file.name}')

if __name__ == '__main__':
    flatten_test_datasets()
