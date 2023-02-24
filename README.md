# Diversely Regularized Matrix Factorization for Accurate and Aggregately Diversified Recommendation

This project is a pytorch implementation of Diversely Regularized Matrix Factorization for Accurate and Aggregately Diversified Recommendation.
This paper proposes a novel approach, Diversely Regularized Matrix Factorization (DivMF), to achieve high aggregate level diversity in recommendation system while maintaining accuracy.
This project provides executable source code with adjustable hyperparameters as arguments and preprocessed datasets which used in the paper.

## Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Scipy](https://scipy.org)
- [Click](https://click.palletsprojects.com/en/7.x/)
- [tqdm](https://tqdm.github.io/)

## Usage
You can run a demo script `demo.sh` that reproduces the experimental results in the paper.
You can change the hyperparameters by modifying arguments of `main.py`.
Unpack zip files in `data` directory to use large datsets: Yelp-15, Gowalla-15, Movielens-10M.

## Datasets

Preprocessed data are included in the `data` directory.

| Name | Users | Items | Interactions | Download |
| --- | ---: | ---: | ---: | --- |
| Yelp-15 | 69,853 | 43,671 | 2,807,606 | [Link](https://www.yelp.com/dataset) |
| Gowalla-15 | 34,688 | 63,729 | 2,438,708 | [Link](https://snap.stanford.edu/data/loc-gowalla.html) |
| Epinions-15 | 5,531 | 4,286 | 186,995 | [Link](http://www.trustlet.org/downloaded_epinions.html) |
| Movielens-10M | 69.878 | 10,677 | 10,000,054 | [Link](https://grouplens.org/datasets/movielens/1m/) |
| Movielens-1M | 6,040 | 3,706 | 1,000,209 | [Link](https://grouplens.org/datasets/movielens/1m/) |
