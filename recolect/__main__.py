import argparse
import pickle
from recolect.core import get_recommendations, load_and_preprocess_data, load_data, train


def save_model(model, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)


def load_model(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def training_cmd(args):
    """Train the recommendation model"""
    model, _ = train(load_and_preprocess_data(args.filepath, args.col))
    save_model(model, args.modelpath)
    print(f"Training complete. Model saved to {args.modelpath}")


def recommend_cmd(args):
    """Get recommendations for a title"""
    model = load_model(args.modelpath)
    data = load_data(args.filepath)
    result = get_recommendations(args.title, model, data, args.n, args.method)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='recolect command line interface')
    subparsers = parser.add_subparsers(dest="command")

    # Train parser
    train_parser = subparsers.add_parser('train', help='Train the recommendation model')
    train_parser.add_argument('--filepath', type=str, required=True, help='Path to the data file')
    train_parser.add_argument('--col', type=str, required=True, help='Column name of the data file')
    train_parser.add_argument('--modelpath', type=str, required=True, help='Path to save the trained model')
    train_parser.set_defaults(func=training_cmd)

    # Recommend parser
    recommend_parser = subparsers.add_parser('recommend', help='Get recommendations for a title')
    recommend_parser.add_argument('title', type=str, help='Title of the item')
    recommend_parser.add_argument('--modelpath', type=str, required=True, help='Path to the trained model')
    recommend_parser.add_argument('--filepath', type=str, required=True, help='Path to the data file')
    recommend_parser.add_argument('--n', type=int, default=10, help='Number of recommendations to return')
    recommend_parser.add_argument('--method', type=str, default="default_method", help='Method to use for recommendations')
    recommend_parser.set_defaults(func=recommend_cmd)

    args = parser.parse_args()

    if args.command is None:
        print("Please specify a command. Use --help for more information.")
    else:
        args.func(args)
