import jsonlines
from tqdm import tqdm
from singularity_nlp.util.translation import translate


def data_translate(input_file, output_file, line_cnt=None, target='ja', source='en'):
    data_ja = []
    with jsonlines.open(input_file, 'r') as fr, jsonlines.open(output_file, 'w') as fw:
        for item in tqdm(fr, total=line_cnt):
            history_ja = []
            flag = True
            for turn in item['history']:
                turn_ja = {}
                for k, v in turn.items():
                    v_ja = translate(v, target=target, source=source, format='text')
                    if v_ja:
                        turn_ja[k] = v_ja
                    else:
                        flag = False        # 标识一下，整个item都要舍弃
                        break
                if flag is False:
                    break
                history_ja.append(turn_ja)

            if flag:
                item_ja = item.copy()
                item_ja['history'] = history_ja
                data_ja.append(item_ja)
                fw.write(item_ja)
    return data_ja



if __name__ == '__main__':
    data_ja = data_translate('mtbench101.jsonl', 'mtbench101_ja.jsonl', line_cnt=1388, target='ja', source='en')
    data_ar = data_translate('mtbench101.jsonl', 'mtbench101_ar.jsonl', line_cnt=1388, target='ar', source='en')
    data_id = data_translate('mtbench101.jsonl', 'mtbench101_id.jsonl', line_cnt=1388, target='id', source='en')
    data_fr = data_translate('mtbench101.jsonl', 'mtbench101_fr.jsonl', line_cnt=1388, target='fr', source='en')
    data_de = data_translate('mtbench101.jsonl', 'mtbench101_de.jsonl', line_cnt=1388, target='de', source='en')
    data_it = data_translate('mtbench101.jsonl', 'mtbench101_it.jsonl', line_cnt=1388, target='it', source='en')
    data_es = data_translate('mtbench101.jsonl', 'mtbench101_es.jsonl', line_cnt=1388, target='es', source='en')
