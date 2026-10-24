"""
Data preprocessing for MovieLens 1M.

Loads raw .dat files, remaps IDs to contiguous indices, encodes genres as
multi-hot vectors, and does a per-user 80/20 train/test split.

Saves pickled tensors ready for the model.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


GENRE_LIST=[
    "Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western",
]
N_GENRES=len(GENRE_LIST)
GENRE_TO_IDX={g:i for i,g in enumerate(GENRE_LIST)}


def load_ratings(data_dir:Path)->pd.DataFrame:
    """Load ratings.dat: UserID::MovieID::Rating::Timestamp"""
    path=data_dir/"ratings.dat"
    df=pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id","movie_id","rating","timestamp"],
        encoding="latin-1",
    )
    return df


def load_movies(data_dir:Path)->pd.DataFrame:
    """Load movies.dat: MovieID::Title::Genres"""
    path=data_dir/"movies.dat"
    df=pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["movie_id","title","genres"],
        encoding="latin-1",
    )
    return df


def load_users(data_dir:Path)->pd.DataFrame:
    """Load users.dat: UserID::Gender::Age::Occupation::Zip-code"""
    path=data_dir/"users.dat"
    df=pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id","gender","age","occupation","zip"],
        encoding="latin-1",
    )
    return df


def encode_genres(genres_str:str)->np.ndarray:
    """Turn 'Action|Adventure' into an 18-dim multi-hot vector."""
    vec=np.zeros(N_GENRES,dtype=np.float32)
    for g in genres_str.split("|"):
        if g in GENRE_TO_IDX:
            vec[GENRE_TO_IDX[g]]=1.0
    return vec


def build_dataset(data_dir:Path,out_dir:Path,seed:int=42,test_frac:float=0.2):
    """Run the full preprocessing pipeline."""
    print(f"[data] Loading raw files from {data_dir}")
    ratings=load_ratings(data_dir)
    movies=load_movies(data_dir)
    users=load_users(data_dir)

    print(
        f"[data] Raw: {len(ratings):,} ratings, "
        f"{ratings.user_id.nunique():,} users, "
        f"{ratings.movie_id.nunique():,} movies"
    )

    # Remap IDs to contiguous 0..N-1
    # Only keep movies that actually appear in ratings
    unique_users=sorted(ratings.user_id.unique())
    unique_movies=sorted(ratings.movie_id.unique())
    user2idx={u:i for i,u in enumerate(unique_users)}
    movie2idx={m:i for i,m in enumerate(unique_movies)}
    idx2user={i:u for u,i in user2idx.items()}
    idx2movie={i:m for m,i in movie2idx.items()}

    ratings["user_idx"]=ratings.user_id.map(user2idx)
    ratings["movie_idx"]=ratings.movie_id.map(movie2idx)

    n_users=len(unique_users)
    n_movies=len(unique_movies)
    print(f"[data] After remapping: {n_users:,} users, {n_movies:,} movies")

    # Build the genre matrix using movie_idx
    genre_matrix=np.zeros((n_movies,N_GENRES),dtype=np.float32)
    movie_title={}
    movie_genres_str={}
    for _,row in movies.iterrows():
        mid=row["movie_id"]
        if mid in movie2idx:
            mi=movie2idx[mid]
            genre_matrix[mi]=encode_genres(row["genres"])
            movie_title[mi]=row["title"]
            movie_genres_str[mi]=row["genres"]

    # Per-user 80/20 split
    print(f"[data] Splitting per-user 80/20 (seed={seed})")
    rng=np.random.RandomState(seed)
    train_rows=[]
    test_rows=[]
    for uid,group in ratings.groupby("user_idx"):
        idx=group.index.to_numpy()
        rng.shuffle(idx)
        n_test=max(1,int(len(idx)*test_frac))
        test_idx=idx[:n_test]
        train_idx=idx[n_test:]
        train_rows.append(train_idx)
        test_rows.append(test_idx)

    train_idx=np.concatenate(train_rows)
    test_idx=np.concatenate(test_rows)

    train=ratings.loc[train_idx,["user_idx","movie_idx","rating","timestamp"]].reset_index(drop=True)
    test=ratings.loc[test_idx,["user_idx","movie_idx","rating","timestamp"]].reset_index(drop=True)

    print(f"[data] Train: {len(train):,}  Test: {len(test):,}")

    # Save everything
    out_dir.mkdir(parents=True,exist_ok=True)
    payload={
        "train":train.to_numpy(),  # columns: user_idx, movie_idx, rating, timestamp
        "test":test.to_numpy(),
        "n_users":n_users,
        "n_movies":n_movies,
        "n_genres":N_GENRES,
        "genre_list":GENRE_LIST,
        "genre_matrix":genre_matrix,  # (n_movies, n_genres)
        "movie_title":movie_title,  # movie_idx -> title
        "movie_genres_str":movie_genres_str,  # movie_idx -> 'Action|Comedy'
        "user2idx":user2idx,
        "movie2idx":movie2idx,
        "idx2user":idx2user,
        "idx2movie":idx2movie,
    }

    out_path=out_dir/"processed.pkl"
    with open(out_path,"wb") as f:
        pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[data] Wrote {out_path}")
    return payload


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./ml-1m",
        help="Directory containing ratings.dat, movies.dat, users.dat",
    )
    parser.add_argument("--out_dir",type=str,default="./data")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--test_frac",type=float,default=0.2)
    args=parser.parse_args()

    build_dataset(
        Path(args.data_dir),
        Path(args.out_dir),
        seed=args.seed,
        test_frac=args.test_frac,
    )