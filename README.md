# entity-classifier
Classify entities into clusters via embedding vectors, using a given list of category names.

- use embedding to make vectors of entity names
- then cluster against a list of words/categories
- then can classify for presentation in logical groups

## Apprroach

Compare words (labels) by examining how close are their encoded vectors:

- the dot product of 2 normalised vectors = cosine Angle
- cosine distance = 1 - v.w
  - smaller means closer

## Dependencies

- Python 3.11
- pyenv - if on Windows use [pyenv-win](https://github.com/pyenv-win/pyenv-win)

## Install

Switch to Python 3.11.6:

```
pyenv install 3.11.6
pyenv local 3.11.6
```

Setup a virtual environment:

```
./create_env.sh
```

Install SBERT and cornsnake via this pip command:

```
pip install -U sentence-transformers==2.2.2 cornsnake==0.0.26
```

## Usage

```
python main.py <path to category list file> <path to entity names file> [threshold (number between 0 and 1)]
```

## Example

To test:

```
./test.sh
```

OUTPUT:

```
CATEGORY: (unknown)
  entity ['Aardvark', 'Alpaca', 'Anaconda']
CATEGORY: animal
  entity ['Albatross', 'Ant', 'Zebu']
CATEGORY: country
  entity ['Alligator', 'Albania', 'Andorra', 'Angola', 'Austria', 'Bangladesh', 'Belgium']
```

The results are not perfect, but not bad considering this is a simple 'out of the box' solution.

## Further improvements

Hierarchy of labels:

- first, classify against a top-level list of labels
- then, for each label, classify against that labels list of sub-labels

Increase accuracy:

- take several embeddings per class and use their average for that class
- try different embeddings, can get better results
- try different distance measures from your library
- consider tuning the embedding

# References

[Conference notes from ML Con Berlin 2023](https://github.com/mrseanryan/dev-conferences/blob/master/2023/mlcon-berlin/talk-Embeddings-Intro.md/README.md)

[SBERT: How to Use Sentence Embeddings to Solve Real-World Problems](https://anirbansen2709.medium.com/sbert-how-to-use-sentence-embeddings-to-solve-real-world-problems-f950aa300c72)
