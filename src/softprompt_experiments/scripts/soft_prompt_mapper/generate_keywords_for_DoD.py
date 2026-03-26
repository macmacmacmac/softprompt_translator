import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

# TODO: Improve this method, as its skipping a lot of hyponyms
def get_related_classes(category_word, num_classes=10):
    # Get synonym sets
    synsets = wordnet.synsets(category_word)
    if not synsets:
        return []
    
    # synsets[0] grabs the most common definition based synonyms
    # Get specific instances (hyponyms) of that category
    hyponyms = synsets[0].hyponyms()
    
    # Clean up the names (replace underscores with spaces)
    classes = [h.lemma_names()[0].replace('_', ' ') for h in hyponyms]
    
    # Return the first `num_classes` examples
    return classes[:num_classes]

if __name__ == '__main__':
    print(f"Related classes for metals: \n{get_related_classes("metals")}\n")      
    print(f"Related classes for professions: \n{get_related_classes("professions")}\n") 
    print(f"Related classes for emotions: \n{get_related_classes("emotions")}\n")
    print(f"Related classes for color: \n{get_related_classes("color")}\n")     

