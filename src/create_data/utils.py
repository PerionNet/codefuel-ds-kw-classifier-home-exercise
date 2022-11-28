import pickle as pkl
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertTokenizerFast
import sklearn
import ast


def get_negative_class_dataset(dataset, label_to_replace=-100, h2_negative_class=218, h3_negative_class=384):
    h2_label = dataset['h2_label']
    h3_label = dataset['h3_label']
    new_h2_label = []
    new_h3_label = []

    for idx, h2_tensor in enumerate(h2_label):
        h2 = h2_tensor.item()
        h3 = h3_label[idx].item()
        if h2 == label_to_replace:
            new_h2_label.append(h2_negative_class)
        else:
            new_h2_label.append(h2)

        if h3 == label_to_replace:
            new_h3_label.append(h3_negative_class)
        else:
            new_h3_label.append(h3)

    new_h2_label = np.array(new_h2_label)
    new_h2_label = torch.from_numpy(new_h2_label)
    new_h3_label = np.array(new_h3_label)
    new_h3_label = torch.from_numpy(new_h3_label)
    dataset['h2_label'] = new_h2_label
    dataset['h3_label'] = new_h3_label
    return dataset


def create_single_mapping_matrix(source_mapping, target_mapping, indices_mapping):
    shape_x = len(source_mapping)
    shape_y = len(target_mapping)
    matrix = torch.zeros(shape_x, shape_y)
    matrix = matrix - float("Inf")
    for key, val in indices_mapping.items():
        val = ast.literal_eval(val)
        matrix[key][val] = 1
    return matrix


def get_mapping_indicator_matrix(h1_mapping_dict, h2_mapping_dict, h3_mapping_dict, h1_h2_mapping_dict,
                                 h2_h3_mapping_dict, is_negative_class):
    h1_category_dict = h1_mapping_dict['category']
    h2_category_dict = h2_mapping_dict['category']
    h3_category_dict = h3_mapping_dict['category']
    h1_h2_mapping_indices_dict = h1_h2_mapping_dict['target_indices']
    h2_h3_mapping_indices_dict = h2_h3_mapping_dict['target_indices']
    h1_h2_indicator_matrix = create_single_mapping_matrix(h1_category_dict, h2_category_dict,
                                                          h1_h2_mapping_indices_dict)
    h2_h3_indicator_matrix = create_single_mapping_matrix(h2_category_dict, h3_category_dict,
                                                          h2_h3_mapping_indices_dict)
    if is_negative_class:
        # h1 h2 mapping
        vec_col = torch.ones(h1_h2_indicator_matrix.shape[0], 1)
        h1_h2_indicator_matrix = torch.cat((h1_h2_indicator_matrix, vec_col), 1)
        # h2 h3 mapping
        vec_row = torch.zeros(1, h2_h3_indicator_matrix.shape[1])
        vec_col = torch.ones(h2_h3_indicator_matrix.shape[0] + 1, 1)
        h2_h3_indicator_matrix = torch.cat((h2_h3_indicator_matrix, vec_row), 0)
        h2_h3_indicator_matrix = torch.cat((h2_h3_indicator_matrix, vec_col), 1)


    return h1_h2_indicator_matrix, h2_h3_indicator_matrix


def get_sampler_weights(dataset, weights_len=27):
    weights_counter = weights_len * [0]
    dataset_len = len(dataset)
    y = []
    for i in range(dataset_len):
        _, _, _, y_h1, _, _, _, _ = dataset[i]
        weights_counter[y_h1] += 1
        y.append(y_h1)
    weight = [1. / t for t in weights_counter]
    samples_weight = np.array([weight[t] for t in y])
    samples_weight = torch.from_numpy(samples_weight)

    return samples_weight


def get_h1_balanced_class_weights(dataset, weights_len=27):
    dataset_len = len(dataset)
    weights_counter = weights_len * [0]
    y = []
    for i in range(dataset_len):
        _, _, _, y_h1, _, _, _, _ = dataset[i]
        weights_counter[y_h1.item()] += 1
        y.append(y_h1.item())
    # weight = [1. / t for t in weights_counter]
    weights_arr = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=list(set(y)), y=y)

    return weights_arr


def parse_taxonomy_dict(taxonomy_ids_path):
    df = pd.read_csv(taxonomy_ids_path)
    df_dict = df.to_dict('list')
    category_id_list = df_dict['CATEGORYID']
    category_list = df_dict['CATEGORY']
    mapping_dict = {}
    for c_id, c in zip(category_id_list, category_list):
        mapping_dict[c] = c_id
    return mapping_dict


# This function maps from hi->hj (h1->h2 / h2 -> h3)
# It returns list of indices
def get_indices_mapping_hi_hj(hi, hj):
    final_mapping_dict = {'source_category': [], 'target_indices': []}
    source_categories = hi['category']
    target_categories = hj['category']
    for source in source_categories:
        indices = []
        for idx, target in enumerate(target_categories):
            if source in target:
                indices.append(idx)
        final_mapping_dict['source_category'].append(source)
        final_mapping_dict['target_indices'].append(indices)
    return final_mapping_dict


def generate_tokenization(data_path, taxonomy_path):
    df = pd.read_csv(data_path)
    taxonomy_dict = parse_taxonomy_dict(taxonomy_path)
    taxonomy_keys = taxonomy_dict.keys()
    df = df.dropna()
    dict_to_save = {'tokens': [], 'h1_label': [], 'h2_label': [], 'h3_label': [], 'category_id': []}
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    searchterm_list = df.searchterm.to_list()
    h1_label = df.h1_label.to_list()
    h2_label = df.h2_label.to_list()
    h3_label = df.h3_label.to_list()
    category = df.category.to_list()
    category_id_list = []
    for cat in category:
        if cat not in taxonomy_keys:
            category_id_list.append(491)
        else:
            category_id_list.append(taxonomy_dict[cat])

    tokenized_dataset = tokenizer(searchterm_list, padding="max_length", max_length=20, truncation=True,
                                  return_tensors='pt')

    dict_to_save['tokens'] = tokenized_dataset
    dict_to_save['h1_label'] = h1_label
    dict_to_save['h2_label'] = h2_label
    dict_to_save['h3_label'] = h3_label
    dict_to_save['category_id'] = category_id_list
    return dict_to_save


def load_tokenization(output_path):
    with open(output_path, 'rb') as f:
        data = pkl.load(f)
    return data


def create_mapping_for_single_head(categories_array, level=1):
    category_arr = []
    idx_arr = []
    current_idx = 0
    for element in categories_array:
        parsed_arg = element.split('/')
        parsed_arg.remove('')
        if len(parsed_arg) == level:
            new_str = ''
            for t in parsed_arg:
                new_str += '{}{}'.format('/', t)
            if new_str not in category_arr:
                category_arr.append(new_str)
                idx_arr.append(current_idx)
                current_idx += 1
    sorted_category = sorted(category_arr)
    return {'category': sorted_category}


def create_mapping_for_categories(input_path, output_dir):
    with open(input_path) as file:
        lines = [line.rstrip() for line in file]

    spreaded_category_array = []
    for line in lines:
        if line == '/General':
            continue
        parsed_arg = line.split('/')
        parsed_arg.remove('')

        ##### SPREADING ######
        tmp_str = ''
        for ele in parsed_arg:
            tmp_str += "{}{}".format('/', ele)
            spreaded_category_array.append(tmp_str)
        ######################

    unique_category_array = list(set(spreaded_category_array))

    level_1_mapping = create_mapping_for_single_head(unique_category_array, level=1)
    level_2_mapping = create_mapping_for_single_head(unique_category_array, level=2)
    level_3_mapping = create_mapping_for_single_head(unique_category_array, level=3)

    h1_h2_mapping = get_indices_mapping_hi_hj(level_1_mapping, level_2_mapping)
    h2_h3_mapping = get_indices_mapping_hi_hj(level_2_mapping, level_3_mapping)

    level_1_df = pd.DataFrame(level_1_mapping)
    level_2_df = pd.DataFrame(level_2_mapping)
    level_3_df = pd.DataFrame(level_3_mapping)

    h1_h2_df = pd.DataFrame(h1_h2_mapping)
    h2_h3_df = pd.DataFrame(h2_h3_mapping)

    level_1_df.to_csv(output_dir + 'level_1_mapping.csv')
    level_2_df.to_csv(output_dir + 'level_2_mapping.csv')
    level_3_df.to_csv(output_dir + 'level_3_mapping.csv')
    h1_h2_df.to_csv(output_dir + 'h1_h2_mapping_indices.csv')
    h2_h3_df.to_csv(output_dir + 'h2_h3_mapping_indices.csv')


def category_arr_to_str(category_arr, level):
    new_str = ""
    if level == 1:
        new_str = "{}{}".format('/', category_arr[0])
    elif level == 2:
        new_str = "{}{}{}{}".format('/', category_arr[0], '/', category_arr[1])
    elif level == 3:
        new_str = "{}{}{}{}{}{}".format('/', category_arr[0], '/', category_arr[1], '/', category_arr[2])

    return new_str


def create_datasets(data_path, h1_mapping_path, h2_mapping_path, h3_mapping_path, out_path):
    df = pd.read_csv(data_path, header=0, usecols=[1, 5], names=['X', 'y']).dropna()
    h1_map = pd.read_csv(h1_mapping_path)
    h1_indexes = list(range(len(h1_map.category)))
    h1_map_dict = dict(zip(h1_map.category, h1_indexes))
    h2_map = pd.read_csv(h2_mapping_path)
    h2_indexes = list(range(len(h2_map.category)))
    h2_map_dict = dict(zip(h2_map.category, h2_indexes))
    h3_map = pd.read_csv(h3_mapping_path)
    h3_indexes = list(range(len(h3_map.category)))
    h3_map_dict = dict(zip(h3_map.category, h3_indexes))

    data = {'searchterm': [], 'h1_label': [], 'h2_label': [], 'h3_label': [], 'category': []}
    X = df.X.to_list()
    new_X = []
    y = df.y.to_list()
    categories_array = []
    y_filtered = []
    for idx, y_i in enumerate(y):
        if y_i == '/General':
            continue
        y_filtered.append(y_i)
        parsed_arg = y_i.split('/')
        parsed_arg.remove('')
        categories_array.append(parsed_arg)
        new_X.append(X[idx])

    for idx, y_i_disentangled in enumerate(categories_array):
        if len(y_i_disentangled) == 1:
            data['searchterm'].append(new_X[idx])
            data['h1_label'].append(h1_map_dict[y_filtered[idx]])
            data['h2_label'].append(-100)
            data['h3_label'].append(-100)
            data['category'].append(y_filtered[idx])
        elif len(y_i_disentangled) == 2:
            new_str = category_arr_to_str(y_i_disentangled, 1)
            data['searchterm'].append(new_X[idx])
            data['h1_label'].append(h1_map_dict[new_str])
            data['h2_label'].append(h2_map_dict[y_filtered[idx]])
            data['h3_label'].append(-100)
            data['category'].append(y_filtered[idx])
        elif len(y_i_disentangled) == 3:
            new_str = category_arr_to_str(y_i_disentangled, 1)
            new_str2 = category_arr_to_str(y_i_disentangled, 2)
            data['searchterm'].append(new_X[idx])
            data['h1_label'].append(h1_map_dict[new_str])
            data['h2_label'].append(h2_map_dict[new_str2])
            data['h3_label'].append(h3_map_dict[y_filtered[idx]])
            data['category'].append(y_filtered[idx])

    new_df = pd.DataFrame(data)
    new_df.to_csv(out_path)
