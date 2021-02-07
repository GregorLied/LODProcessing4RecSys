LODProcessing4RecSys
==================

In this repository we report the steps necessary to leverage knowledge graphs as side information for recommendation.

### Preprocessing

We built our dataset based on the mappings for the items in [Movielens1M](https://grouplens.org/datasets/movielens/1m/) and [LastFM](http://www.lastfm.com}) to DBPedia entities provided in [1]. Thereby, we apply the following processing steps:

**Step 1: Correct DBPedia Mappings**<br>
For each recommendation dataset, we use `[RS] 1. Correct_Mappings.ipynb` to correct the DBPedia mappings.  We obtain `Mapping2DBpedia-1.2-corrected.tsv` as an output. In total, we mapped 3,244 out of a total of 3,883 movies in Movielens1M and 8,790 out of 17,632 artists in LastFM.

The corrections were made using the following steps:

* We check if the provided mappings contain obvious errors.<br>
* We correct the errors using a simple algorithm by generating a list of possible URIs and find the URIs that seems to be a valid mapping for that item. We delete all errors that can't be corrected.<br>
* If multiple items are mapped to the same entity, we only keep one of the item, as the mappings have to be unique.<br>

**Step 2: Get Wikidata Mappings**<br>
For each recommendation dataset, given the corrected mappings of the items to DBpedia entities, we use `[RS] 2.GetWikidata` to exploit the owl:sameAs relation in order to get the corresponding mappings of the items to Wikidata entities. We obtain `Mapping2Wikidata-1.2-corrected.tsv` as an output. Here, we are also able to map 3,244 out of a total of 3,883 movies in Movielens1M and 8,790 out of 17,632 artists in LastFM.

**Step 3: Create Knowledge Graph**<br>
For each recommendation dataset, we then use the URIs in `Mapping2DBpedia-1.2-corrected.tsv` and `Mapping2Wikidata-1.2-corrected.tsv` as seeds and use `[RS] 3.ExpandKG` to apply a BFS with depth 2 for each seed on the rdf dumps in order to generate the triples of the corresponding knowledge graph. We obtain the knowledge graph with literals (`2HopsWikidata.nt`, `2HopsDBPedia.nt`) and without literals (`2HopsDBPediaNoLiterals.nt`, `2HopsWikidataNoLiterals.nt`).

**Step 4: Preprocess Data**<br>
Finally, we are now able to preprocess the data to generate our dataset using `preprocess_data.py`.

Thereby, we encode all content of the datasets [2], [3], [4] and obtain corresponding numerical values (indices).
Furthermore, we have to transform the explicit feedback of [Movielens1M](https://grouplens.org/datasets/movielens/1m/) into implicit feedback and perform a datasplit for [2].

[1] https://github.com/sisinflab/LODrecsys-datasets<br>
[2] Recommendation Dataset (`ratings.dat`, `user_artists.dat`)<br>
[3] Knowledge Graph Dataset (`2HopsDBPediaNoLiterals.nt`, `2HopsWikidataNoLiterals.nt`)<br>
[4] Mappings File (`Mapping2DBpedia-1.2-corrected.tsv`, `Mapping2Wikidata-1.2-corrected.tsv`)]


### Final Raw Datset

After applying `Step 4: Preprocess Data`, we obtain the following files:

* `train.txt`, `eval.txt`, `test.txt`
  * Train, Eval and Test file. 
  * Each line represents a certain user with her/his positive interactions with items: (`our_user_id`, `list of our_item_id`).
  * Note that these files only report the positive interactions. During evaluation we then treat all unobserved interactions as negative instances.
* `kg_final.txt`
  * Knowledge Graph file. 
  * Each line represents a (`our_entity_id`, `our_relation_id`, `our_entity_id`) triple.

Furthermore, we provide the following files to get insights into the semantics of our data:
* `user_list.txt`
  * User file. Only used to get the semantics of the users in `train.txt`, `eval.txt`, `test.txt`.
  * Each line represents a certain user with a (`orignal_user_id`, `our_user_id`) tuple, where `our_user_id` is the ID of
    the user in our dataset.
* `item_list.txt`
  * Item file. Only used to get the semantics of the items in `train.txt`, `eval.txt`, `test.txt` and `kg_final.txt`.
  * Each line represents a certain item with a (`original_item_id`, `our_item_id`, `URI`) triple, where `our_item_id` is the ID of the item in our dataset.
* `entity_list.txt`
  * Entity file. Only used to get the semantics of the entities in `kg_final.txt`.
  * Each line represents a certain entity with a (`URI`, `our_entity_id`) tuple, where `our_entity_id` is the ID of the entity in our dataset.
  * Note that the first `n_items` entities are items.
* `relation_list.txt`
  * Relation file. Only used to get the semantics of the relations in `kg_final.txt`.
  * Each line represents a certain relation with a (`URI`, `our_relation_id`) tuple, where `our_relation_id` is the ID of the relation in our dataset.
* `ratings_final.txt`
  * Positive Interactions file. Only reported for the sake of completeness, as it combines the same information as `train.txt`, `eval.txt`, `test.txt` but combined and in a different format.
  * Each line represents a certain positive interaction with a (`our_user_id`, `our_item_id`, `1`) triple, where `1` is the label for a positive interaction.

### Final Processed Datset

We can then use the `InMemoryDataset` of Pytorch Geometric to process and load the final dataset, which is then used by the model. For information about the instances in the processed dataset, please see `src/datasets/knowledgeawarerecommendation.py`.

If you want to apply a different preprocessing scheme to the data, it might be the best start to make changes to this class. In most cases that might be sufficient. If not, changes to `preprocess_data.py` in `Step 4: Preprocess Data` have to be made. Please note that in, the following data conditions have to hold for `InMemoryDataset`: 
* The raw IDs in our dataset have to be mapped to [0, n_instance]. Thereby, no ID should be missing (e.g. 0,1,2,3 and not 0,2,3). Please note that those two requirements are important to have a well-defined embedding table. In `knowledgeawarerecommendation.py` we then remap the items/entities with n_users, so that we have again well-defined indices for the collaborative knowledge graph. All data instances for KGAT model contain those remaped items.
* No duplicate triples in kg_file and no duplicate tuples in rating_file are present.
* Each user in test and eval set has to also be contained in train set


Acknowledgements
==================
You can freely use the data in your own research work, please cite the following paper as reference:

The user must acknowledge the following rules while using this dataset.
* If you apply the data in your own research work, please cite the mentioned paper above as reference.
* Please don't redistribute the data without our permission.
* Please don't use the data for any commercial purposes without our permission.