# -*- coding: utf-8 -*-
import rdflib
import numpy as np
import pandas as pd

import argparse
import gzip
from collections import Counter

class RDFGraphReader:

    def __init__(self, file):
        # Create a Graph
        self.graph = rdflib.Graph()

        # Parse in an RDF file, adding triples to the graph
        if file.endswith('nt.gz'):
            with gzip.open(file, 'rb') as f:
                self.graph.parse(file=f, format='nt')
        else:
            self.graph.parse(file, format=rdflib.util.guess_format(file))

        # Get frequency of all predicate/relation-types in the graph
        self.freq = Counter(self.graph.predicates())

    def __len__(self):
        return len(self.graph)

    def freq(self, relation):
        """ The frequency of a given relation (how many distinct triples does it occur in?)
        Args:
            relation (str): name of the relation
        Returns:
            relation_freq (int): number of distinct triples in the graph with relation.
        """

        if relation not in self.freq:
            relation_freq =  0
        else:
            relation_freq = self.freq[relation]

        return relation_freq

    def triples(self, relation=None):
        """ Yield all triples with a given relation. If relation=None yield all triples in the graph.
        Args:
            relation (str): name of the relation
        Returns:
            s (rdflib.term.URIRef): subject entity (head)
            p (rdflib.term.URIRef): predicate (relation)
            o (rdflib.term.URIRef): object entity (tail)
        """
        for s, p, o in self.graph.triples((None, relation, None)):
            yield s, p, o

    def subjectSet(self):
        """ Get set of all subject entities (heads)
        Args:
            None
        Returns:
            set : set of all subject entities (heads)
        """
        return set(self.graph.subjects())

    def objectSet(self):
        """ Get set of all object entities (tails)
        Args:
            None
        Returns:
            set : set of all object entities (tails)
        """
        return set(self.graph.objects())

    def relationList(self):
        """ Returns a list of relations, ordered by frequency in descending order
        Args:
            None
        Returns:
            res (list) : list of relations, ordered by frequency in descending order
        """
        res = list(set(self.graph.predicates()))
        res.sort(key=lambda rel: - self.freq[rel])
        return res

# Parameters to be selected according to args inputs
DATA_STATISITCS = dict({'movielens': {'n_users': 6040, 'n_items': 3883,'n_ratings': 1000209}, 
                        'lastfm': {'n_users': 1892, 'n_items': 17632,'n_ratings': 92834}})
RATING_FILE_NAME = dict({'movielens': 'ratings.dat', 
                         'lastfm': 'user_artists.dat'})
SEP = dict({'movielens': '::', 
            'lastfm': '\t'})
THRESHOLD = dict({'movielens': 4, 
                  'lastfm': 0})

MAPPING_FILE_NAME = dict({'wikidata': 'Mapping2Wikidata-1.2-corrected.tsv', 
                          'dbpedia': 'Mapping2DBpedia-1.2-corrected.tsv'})
GRAPH_FILE_NAME = dict({'wikidata': '2hopsWikidataNoLiterals.nt', 
                        'dbpedia': '2hopsDBpediaNoLiterals.nt'})


def load_mappings():
    """ Load (Item -> Entity) Mappings from file; 
        Create (Item -> Index <- Entity) Mappings and store to item_list.txt;
        Start to create (Entity -> Index) Mappings; Will be continued in prepare_kg()
    Args:
        None
    Returns:
        None
    """

    print("Step 1 – Load Item-Entity Mappings...")

    # Read Item-Entity Mappings from file
    # Note that the URIs have to be unique (no duplicates), otherwise we can't perform mappings successfully
    mapping_file = './' + DATASET + '/' + MAPPING_FILE_NAME[KG]
    mapping_df = pd.read_csv(mapping_file, sep="\t", header = None, encoding='utf8', engine='python') 

    # Create (Item -> Index) Mappings
    # Start to create (Entity -> Index) Mappings; Will be continued in prepare_kg()
    for index, (item_index_old, entity_id_old) in enumerate(zip(mapping_df[0], mapping_df[2])):
        item_id2index[item_index_old] = index
        entity_id2index[entity_id_old] = index

    # KGAT item_list.txt
    index2entity = {index: entity_id for entity_id, index in entity_id2index.items()}
    with open('./' + DATASET + '/' + KG + '/item_list.txt', 'w', encoding='utf-8') as item_list:
        for item_index_old, index in item_id2index.items():
            item_list.write(f'{item_index_old}\t{index}\t{index2entity[index]}\n')

def prepare_kg():
    """ Download and read KG from RDF-file;
        Store (head relation tail)-triples in kg_final.txt; 
        Complete (Entity -> Index) Mappings and store to entity_list.txt;
        Create (Item -> Index) Mappings and store to relation_list.txt;
        Get statistics about the KG;
    Args:
        None
    Returns:
        None
    """

    print('Step 2 – Prepare Knowledge Graph...')

    print('Get Knowledge Graph')
    graph_file = './' + DATASET + '/' + GRAPH_FILE_NAME[KG]

    print("Reading Knowledge Graph from file")
  
    # Create rdflib.Graph object
    knowledgeGraph = RDFGraphReader(graph_file)

    print('Converting Knowledge Graph...')
    entity_count = len(entity_id2index)
    relation_count = 0
    # Write kg_final.txt (KGAT)
    with open('./' + DATASET + '/' + KG + '/kg_final.txt', 'w', encoding='utf-8') as kg_final:
        for head_old, relation_old, tail_old in knowledgeGraph.triples():

            # transform head_old, relation_old, tail_old rdflib.term.URIRef objects in strings
            head_old = str(head_old)
            relation_old = str(relation_old)
            tail_old = str(tail_old)

            # Complete (Entity -> Index) Mappings
            # Note that we started to do this already in load_mapping()
            # to make sure that the corresponding KG entities to our items have the smallest indices
            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_count
                entity_count += 1
            head = entity_id2index[head_old]

            # Create (Relation -> Index) Mappings
            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_count
                relation_count += 1
            relation = relation_id2index[relation_old]

            # Complete (Entity -> Index) Mappings
            # Note that we started to do this already in load_mapping()
            # to make sure that the corresponding KG entities to our items have the smallest indices
            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_count
                entity_count += 1
            tail = entity_id2index[tail_old]

            kg_final.write(f'{head}\t{relation}\t{tail}\n')

    # Write entity_list.txt (KGAT)
    with open('./' + DATASET + '/' + KG + '/entity_list.txt', 'w', encoding='utf-8') as entity_list:
        for entity_id, index in entity_id2index.items():
            entity_list.write(f'{entity_id}\t{index}\n')
            
    # Write relation_list.txt (KGAT)
    with open('./' + DATASET + '/' + KG + '/relation_list.txt', 'w', encoding='utf-8') as relation_list:
        for relation_id, index in relation_id2index.items():
            relation_list.write(f'{relation_id}\t{index}\n')
    
    # Print some Knowledge Graph Statistics
    n_entities = entity_count
    n_relations = relation_count
    n_triples = len(knowledgeGraph)
    print("===============================================")
    print("       Statistics about Knowledge Graph        ")
    print("===============================================")
    print(f"{'Number of Entities:': <35} {n_entities:>10}")
    print(f"{'Number of Relations:': <35} {n_relations:>10}")
    print(f"{'Number of Triples:': <35} {n_triples:>10}")
    print("===============================================")

def prepare_ratings():
    """ Getting positive User-Item Ratings according to Threshold;
        Sample for each positive User-Item Rating an unrated item and store to ratings_final.txt;
        Create (User -> Index) Mappings and store to user_list.txt;
        Get Statistics about the original and our rating dataset;
    Args:
        None
    Returns:
        None
    """

    print('Step 3 – Prepare User-Item Ratings...')

    print('Reading User-Item Ratings')
    rating_file = './' + DATASET + '/' + RATING_FILE_NAME[DATASET]
    rating_df = pd.read_csv(rating_file, sep=SEP[DATASET], header = None, encoding='utf8', engine='python') 

    item_set = set(item_id2index.values())
    user_positive_ratings = dict()
    user_negative_ratings = dict()

    rating_count = 0
    for user_id_old, item_index_old, rating in zip(rating_df[0], rating_df[1], rating_df[2]):

        # If no entity mapping for item_index_old has been found skip this item rating
        if item_index_old not in item_id2index: 
            continue

        # Analyze rating, as entity mapping for item_index_old has been found
        item_index = item_id2index[item_index_old]
        rating_count += 1

        # Annotate rating in positive and negative interactions according to a defined threshold
        if rating >= THRESHOLD[DATASET]:
            if user_id_old not in user_positive_ratings:
                user_positive_ratings[user_id_old] = set()
            user_positive_ratings[user_id_old].add(item_index)
        else:
            if user_id_old not in user_negative_ratings:
                user_negative_ratings[user_id_old] = set()
            user_negative_ratings[user_id_old].add(item_index)

    print('Converting User-Item Ratings')
    user_count = 0
    
    """
    # Write ratings_final.txt (KGAT)
    with open('./' + DATASET + '/' + KG + '/ratings_final.txt', 'w', encoding='utf-8') as ratings_final:
        for user_id_old, pos_item_set in user_positive_ratings.items():

            # Create (user-id (original) -> user-id (ours)) Mappings
            if user_id_old not in user_id2index:
                user_id2index[user_id_old] = user_count
                user_count += 1
            user_id = user_id2index[user_id_old]

            for item_index in pos_item_set:
                ratings_final.write(f'{user_id}\t{item_index}\t1\n')
            # Was in original RippleNet, but is not needed here
            #unwatched_set = item_set - pos_item_set
            #if user_id_old in user_negative_ratings:
            #    unwatched_set -= user_negative_ratings[user_id_old]
            #for item_index in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            #    ratings_final.write(f'{user_id}\t{item_index}\t0\n')
    """

    # Write user_list.txt (KGAT)
    with open('./' + DATASET + '/' + KG + '/user_list.txt', 'w', encoding='utf-8') as user_list:
        for user_id, index in user_id2index.items():
            user_list.write(f'{user_id}\t{index}\n')

    # Get some statistics about the Bipartite Graph
    n_users_original = DATA_STATISITCS[DATASET]['n_users']
    n_items_original = DATA_STATISITCS[DATASET]['n_items']
    n_ratings_original = DATA_STATISITCS[DATASET]['n_ratings']
    sparsity_original = (1.0 - (n_ratings_original * 1.0)/(n_users_original * n_items_original))*100

    n_user_ours = user_count
    n_items_ours = len(item_set)
    n_ratings_ours = rating_count
    sparsity_ours = (1.0 - (n_ratings_ours * 1.0)/(n_user_ours * n_items_ours))*100
    n_positive_ratings = sum(len(pos_item_set) for pos_item_set in user_positive_ratings.values())

    print("===============================================")
    print("       Statistics about original Dataset       ")
    print("===============================================")
    print(f"{'Number of User:': <35} {n_users_original:>10}")
    print(f"{'Number of Items:': <35} {n_items_original:>10}")
    print(f"{'Number of Interactions:': <35} {n_ratings_original:>10}")
    print(f"{'Data Sparsity:': <35} {sparsity_original:>10.2f}")
    # Note that the decrease in users for movielens comes from the fact that some users have no recorded positive item interaction.
    print("===============================================")
    print("          Statistics about our Dataset         ")  
    print("===============================================")
    print(f"{'Number of User:': <35} {n_user_ours:>10}")
    print(f"{'Number of Items:': <35} {n_items_ours:>10}")
    print(f"{'Number of Interactions:': <35} {n_ratings_ours:>10}")
    print(f"{'Data Sparsity:': <35} {sparsity_ours:>10.2f}")
    print(f"{'Positive Ratings:': <35} {n_positive_ratings:>10}")
    print(f"{'': <19} ({n_positive_ratings/n_ratings_ours:.4f}% of all Interactions)")
    print("===============================================")

def split_data_for_kgat(eval_ratio, test_ratio):
    """ Reloading ratings_final.txt created by prepare_ratings();
        Perform Data Split and save to train.txt and test.txt;
    Args:
        eval_ratio (float): Ratio for eval set
        test_ratio (float): Ratio for test set
    Returns:
        None
    """

    print('Step 4 – Split User-Item Ratings...')

    ratings_final_file = './' + DATASET + '/' + KG + '/ratings_final.txt'
    rating_np = np.loadtxt(ratings_final_file, dtype=np.int32)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    n_ratings = rating_np.shape[0]

    # Allocate row indices in rating_np (ndarray) to train, eval and test dataset
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    print(f"Size of Train, Eval, Test set after split with Train: {int((1-eval_ratio-test_ratio)*100)}% | Eval: {int(eval_ratio*100)}% | Test: {int(test_ratio*100)}%:")
    print(len(train_indices), len(eval_indices), len(test_indices))

    # Traverse training data, only keeping the users with positive ratings in training set
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    # Remove all users that have no positive ratings from eval and test set
    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    print("Size of Train, Eval, Test set w/o all users with no positive rating in Train set:")
    print(len(train_indices), len(eval_indices), len(test_indices))

    # Save Data Accordingly
    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    train_dict = {}
    for row in train_data:
        user = row[0]
        item = row[1]
        rating = row[2]

        if rating == 1:
            if user not in train_dict:
                train_dict[user] = []
            train_dict[user].append(item)

    eval_dict = {}
    for row in eval_data:
        user = row[0]
        item = row[1]
        rating = row[2]

        if rating == 1:
            if user not in eval_dict:
                eval_dict[user] = []
            eval_dict[user].append(item)

    test_dict = {}
    for row in test_data:
        user = row[0]
        item = row[1]
        rating = row[2]

        if rating == 1:
            if user not in test_dict:
                test_dict[user] = []
            test_dict[user].append(item)

    # Write train.txt (KGAT)
    with open('./' + DATASET + '/' + KG + '/train.txt', 'w', encoding='utf-8') as train:
        for user, pos_item_set in train_dict.items():
            item_string = ' '.join(str(item_id) for item_id in pos_item_set)
            train.write(f'{user} {item_string}\n')

    # Write eval.txt (KGAT)
    with open('./' + DATASET + '/' + KG + '/eval.txt', 'w', encoding='utf-8') as val:
        for user, pos_item_set in eval_dict.items():
            item_string = ' '.join(str(item_id) for item_id in pos_item_set)
            val.write(f'{user} {item_string}\n')

    # Write test.txt (KGAT)
    with open('./' + DATASET + '/' + KG + '/test.txt', 'w', encoding='utf-8') as test:
        for user in sorted(test_dict.keys()):
            item_string = ' '.join(str(item_id) for item_id in test_dict[user])
            test.write(f'{user} {item_string}\n')

def dataset_split(eval_ratio, test_ratio):
    """ Reloading ratings_final.txt created by prepare_ratings();
        Perform Data Split and save to train.txt, eval.txt and test.txt;
    Args:
        eval_ratio (float): Ratio for eval set
        test_ratio (float): Ratio for test set
    Returns:
        None
    """

    print('Step 4 – Split User-Item Ratings...')

    ratings_final_file = './' + DATASET + '/' + KG + '/ratings_final.txt'
    rating_np = np.loadtxt(ratings_final_file, dtype=np.int32)

    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    print(f"Size of Train, Eval, Test set after split with Train: {int((1-eval_ratio-test_ratio)*100)}% | Eval: {int(eval_ratio*100)}% | Test: {int(test_ratio*100)}%:")
    print(len(train_indices), len(eval_indices), len(test_indices))

    # Traverse training data, only keeping the users with positive ratings in training set
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    # Remove all users that have no positive ratings from eval and test set
    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    print("Size of Train, Eval, Test set w/o all users with no positive rating in Train set:")
    print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]
    
    with open('./' + DATASET + '/' + KG + '/train.txt', 'w', encoding='utf-8') as train_file:
        for i in range(train_data.shape[0]):
            user = train_data[i][0]
            item = train_data[i][1]
            rating = train_data[i][2]
            train_file.write(f'{user}\t{item}\t{rating}\n')

    with open('./' + DATASET + '/' + KG + '/eval.txt', 'w', encoding='utf-8') as eval_file:
        for i in range(eval_data.shape[0]):
            user = eval_data[i][0]
            item = eval_data[i][1]
            rating = eval_data[i][2]
            eval_file.write(f'{user}\t{item}\t{rating}\n')

    with open('./' + DATASET + '/' + KG + '/test.txt', 'w', encoding='utf-8') as test_file:
        for i in range(test_data.shape[0]):
            user = test_data[i][0]
            item = test_data[i][1]
            rating = test_data[i][2]
            test_file.write(f'{user}\t{item}\t{rating}\n')

if __name__ == '__main__':
    
    # Example Usage: 
    # python prepare_dataset.py --dataset movielens --kg dbpedia --eval_ratio 0.1 --test_ratio 0.2
    # python prepare_dataset.py --dataset lastfm --kg wikidata --eval_ratio 0.1 --test_ratio 0.2

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2021, help='Random seed')
    parser.add_argument('--dataset', type=str, default='movielens', help='Choose dataset from {movielens, lastfm}')
    parser.add_argument('--kg', type=str, default='wikidata', help='Choose knowledge graph from {wikidata, dbpedia}')
    parser.add_argument('--eval_ratio', type=float, default=0.1, help='Ratio for eval set.')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio for test set.')
    
    args = parser.parse_args()

    np.random.seed(args.seed)
    DATASET = args.dataset
    KG = args.kg
    eval_ratio = args.eval_ratio
    test_ratio = args.test_ratio
    
    assert DATASET in ['movielens', 'lastfm']
    assert KG in ['wikidata', 'dbpedia']

    print("Start Preprocessing...")
    item_id2index = dict()
    entity_id2index = dict()
    relation_id2index = dict()
    user_id2index = dict()

    load_mappings()
    prepare_kg()
    prepare_ratings()
    #For KGAT:
    split_data_for_kgat(eval_ratio,test_ratio)
    #For RippleNet:
    #dataset_split(eval_ratio, test_ratio)

    print('Preprocessing done.')
