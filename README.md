# recolect

A toolbox for smart suggestions-making tools, making it easier to create your own recommendation systems.

Train models on your dataset and get recommendations using a simple command-line interface.

## Installation

```
pip install recolect
```

## Structure

```
recolect/
│
├── core.py  (contains main functions for recommendations)
│
├── __main__.py  (command-line interface)
│
├── ...
```

## Usage

### Training the Model

to train your recommendation model:

```
python -m recolect train --filepath=PATH_TO_YOUR_DATA --col=COLUMN_NAME --modelpath=PATH_TO_SAVE_MODEL
```

`--filepath`: The path to your data file.
`--col`: The column name you're focusing on in your data.
`--modelpath`: The path where you want to save your trained model.

For example:

```
python -m recolect train --filepath=data.csv --col=title --modelpath=model.pkl
```

This command will train your model on the provided data and save it to model.pkl.

## Get Recommendations

To get recommendations using your trained model:

```
python -m recolect recommend "ITEM_TITLE" --modelpath=PATH_TO_TRAINED_MODEL --n=NUMBER_OF_RECOMMENDATIONS --method=RECOMMENDATION_METHOD
```

`ITEM_TITLE`: The title of the item you want recommendations for.

`--modelpath`: The path to your pre-trained model.

`--n`: (Optional) The number of recommendations you want. Default is 10.

`--method`: (Optional) The recommendation method you want to use. Default is "default_method".

For Example

```
python -m recolect recommend "The Shawshank Redemption" --modelpath=model.pkl --n=5 --method=some_method
```

This command will provide 5 recommendations based on "The Shawshank Redemption" using the method some_method.
