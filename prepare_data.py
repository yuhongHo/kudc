import json

def data_precessing(data_dir, save_dir):
    # 对原始数据格式进行预处理，也可以自行在生成知识文件时直接处理成output格式的json文件
    # raw kg data format: utterance	label	entity1:[type1,type2];entity2:[type1,type2]
    # output {'utterance': '开www.china.com', 'label': 'website', 'entity_types': {'www': [], '中国': [0, 1, 2, 3, 4]}, 'type_list': ['历史', '国家', '文明古国', '古国', '国']}
    with open(data_dir) as fo:
        items = fo.readlines()
        data = []
        for item in items:
            data_item = {}
            item = item.strip().split('\t')
            data_item['utterance'], data_item['label'] = item[:2]
            entity_types = item[2] if len(item)==3 else None
            type_list = []
            entities = []
            types = []
            if entity_types:
                entities = [entity_type.split(":")[0] for entity_type in entity_types.split(';')]
                type_tmp = [entity_type.split(":")[1] for entity_type in entity_types.split(';')]
                for t in type_tmp:
                    t = t.replace('[', '').replace(']', '').split(',') if t != '[]' else []
                    types.append(t)
                    type_list += t
            # if entity_types:
            #     entities = entity_types.split(';')[::2]
            #     for t in entity_types.replace(':','').split(' ')[1::2]:
            #         t = t.replace('[','').replace(']','').split(',') if t != '[]' else []
            #         types.append(t)
            #         type_list += t
            assert len(entities) == len(types)
            type_list = list(set(type_list))
            entity_types_pair = {}
            for entity,type in zip(entities,types):
                type_index = [type_list.index(t) for t in type] if type else []
                type_index.sort()
                entity_types_pair[entity] = type_index
            data_item['entity_types'] = entity_types_pair
            data_item['type_list'] = type_list
            data.append(data_item)
            # print(data_item)

    with open(save_dir,'w') as fw:
        json.dump(data,fw)

if __name__ == '__main__':
    smp_train_dir = './data/smp/0_train.csv'
    train_save_dir = 'data/smp/0_train.txt'
    data_precessing(smp_train_dir, train_save_dir)

    smp_valid_dir = './data/smp/0_dev.csv'
    valid_save_dir = 'data/smp/0_dev.txt'
    data_precessing(smp_valid_dir, valid_save_dir)

    smp_test_dir = './data/smp/test.csv'
    test_save_dir = 'data/smp/test.txt'
    data_precessing(smp_test_dir, test_save_dir)