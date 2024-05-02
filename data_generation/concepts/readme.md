# Concept Expansion
This module contains code for expanding the concepts with GPT-3 (`01_extend_concepts.py`)
and converting them to a pickle file (`02_create_concept_dictionary.py`).
The files have to be run in the order specified for the final results.
The pickle output is used in other parts for data generation.
In order to use this module you need access to the list of concepts in your corpus in a csv file.

Both the concept list and the extended dictionary are available in our resource files. 

Here, we provide samples for running the each script:

`01_extend_concepts.py`: 
* `data-type`: either finance or clinical to select the correct prompt
* `input-file `: a csv file that has a column `keyword` indicating the concepts
* `api-key`: openai api key 
* `output-file`: path to the csv file to store the expansions
```
python 01_extend_concepts.py --data-type finance --input-file ../../data/finance/train_concepts_units_extended.csv  --api-key aweripip3j --output-file ../../data/finance/concepts_extended.csv
```


`02_create_concept_dictionary.py`:

* `data-type`: either finance or clinical to select the correct prompt
* `input-file `: a csv file that has the concepts and extensions in `extension` column
* `output-file`: a path to output a pickle file including all the extensions in a dictionary
```
python 02_create_concept_dictionary.py --data-type finance --input-file ../../data/finance/concepts_extended.csv  --output-file ../../data/finance/concepts_to_extentions.pickle
```