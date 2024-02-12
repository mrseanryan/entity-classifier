import sys

from sentence_transformers import SentenceTransformer, util
from cornsnake import util_file

DEFAULT_THRESHOLD = 0.4
model = None

def _encode_list(names):
    names_to_encoding = dict()
    for name in names:
        encoding = model.encode(name)
        names_to_encoding[name] = encoding
    return names_to_encoding

def _find_closest_category(entity_encoding, category_to_encoding, threshold):
    best_category = "(unknown)"
    best_match = 1
    for category in category_to_encoding:
        cos_sim = 1 - util.cos_sim(entity_encoding, category_to_encoding[category])
        if cos_sim < threshold and cos_sim < best_match:
            best_category = category
            best_match = cos_sim
    return best_category

def classify(entities, categories, threshold):
    global model
    if not model:
        #Load the model(here we use minilm)
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # encode the categories and entities
    category_to_encoding = _encode_list(categories)
    entities_to_encoding = _encode_list(entities)
 
    # build a dictionary of classified entities, by comparing the encodings to find the closest category for each entity
    category_to_entities = dict()

    # Compare using the cosine similarity score between each
    for entity in entities:
        category = _find_closest_category(entities_to_encoding[entity], category_to_encoding, threshold)
        if category not in category_to_entities:
            category_to_entities[category] = []
        category_to_entities[category].append(entity)
    return category_to_entities

def _print_usage_and_exit():
    print(f"USAGE: {sys.argv[0]} <path to category list file> <path to entity names file> [threshold (number between 0 and 1)]")
    exit(42)

def _validate_threshold(threshold):
    message = 'threshold must be a number > 0 and < 1'
    if threshold < 0:
        raise ValueError(message)
    if threshold >= 1:
        raise ValueError(message)

if __name__ == '__main__':
    threshold = DEFAULT_THRESHOLD
    if len(sys.argv) not in [3, 4]:
        _print_usage_and_exit()
    path_to_category_list_file = sys.argv[1]
    path_to_entity_names_file = sys.argv[2]
    if len(sys.argv) == 4:
        threshold = float(sys.argv[3])

    _validate_threshold(threshold)

    categories = util_file.read_lines_from_file(path_to_category_list_file, skip_comments = True)
    entity_names = util_file.read_lines_from_file(path_to_entity_names_file, skip_comments = True)
    category_to_entities = classify(entity_names, categories, threshold)
    for category in category_to_entities:
        print(f'CATEGORY: {category}')
        print(f'  entity {category_to_entities[category]}')
