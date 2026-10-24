"""
Evaluation module.

Computes all metrics required by the project:
  - Rating prediction: MAE, RMSE
  - Top-N recommendation: Precision@K, Recall@K, F1@K, NDCG@K

Top-N protocol:
  For each user:
    - Score all items with the trained model
    - Mask out items the user already rated in the TRAIN set (we never
      recommend something they've already seen)
    - Take the top-K remaining items as the recommendation list
    - Ground-truth "relevant" items for this user are their TEST ratings
      (we use all test items here; some papers use only test items with
      rating >= 4 — we report both for thoroughness)
  Then average Precision/Recall/NDCG across users.

This file is designed to be called programmatically from train.py OR from
the command line:
    python -m evaluation.evaluate --ckpt checkpoints/hybrid_ncf.pt --data data/processed.pkl
"""
import argparse
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.hybrid_ncf import HybridNCF

def mae(y_true:np.ndarray,y_pred:np.ndarray)->float:
    return float(np.mean(np.abs(y_true-y_pred)))

def rmse(y_true:np.ndarray,y_pred:np.ndarray)->float:
    return float(np.sqrt(np.mean((y_true-y_pred)**2)))

def precision_at_k(recommended:list,relevant:set,k:int)->float:
    if k==0:
        return 0.0
    topk=recommended[:k]
    hits=sum(1 for item in topk if item in relevant)
    return hits/k

def recall_at_k(recommended:list,relevant:set,k:int)->float:
    if len(relevant)==0:
        return 0.0
    topk=recommended[:k]
    hits=sum(1 for item in topk if item in relevant)
    return hits/len(relevant)

def f1_from_pr(p:float,r:float)->float:
    if p+r==0:
        return 0.0
    return 2*p*r/(p+r)

def ndcg_at_k(recommended:list,relevant:set,k:int)->float:
    topk=recommended[:k]
    dcg=0.0
    for i,item in enumerate(topk):
        if item in relevant:
            dcg+=1.0/np.log2(i+2)
    ideal_hits=min(k,len(relevant))
    if ideal_hits==0:
        return 0.0
    idcg=sum(1.0/np.log2(i+2) for i in range(ideal_hits))
    return dcg/idcg

@torch.no_grad()
def compute_rating_metrics(model,test_arr,genre_matrix,device,batch_size=4096):
    model.eval()
    users=torch.from_numpy(test_arr[:,0].astype(np.int64))
    items=torch.from_numpy(test_arr[:,1].astype(np.int64))
    ratings=test_arr[:,2].astype(np.float32)
    genre_t=torch.from_numpy(genre_matrix).to(device)

    preds=np.empty(len(users),dtype=np.float32)
    for start in range(0,len(users),batch_size):
        end=min(start+batch_size,len(users))
        u=users[start:end].to(device)
        i=items[start:end].to(device)
        g=genre_t[i]
        p=model(u,i,g).cpu().numpy()
        preds[start:end]=np.clip(p,1.0,5.0)

    return {
        "MAE":mae(ratings,preds),
        "RMSE":rmse(ratings,preds),
    },preds

@torch.no_grad()
def compute_topn_metrics(
    model,
    train_arr,
    test_arr,
    genre_matrix,
    n_movies,
    device,
    k:int=10,
    max_users:int=None,
    relevance_threshold:float=None,
    verbose:bool=True,
):
    model.eval()

    train_by_user=defaultdict(set)
    for u,i,_,_ in train_arr:
        train_by_user[int(u)].add(int(i))

    test_by_user=defaultdict(list)
    for u,i,r,_ in test_arr:
        test_by_user[int(u)].append((int(i),float(r)))

    user_ids=sorted(test_by_user.keys())
    if max_users is not None:
        user_ids=user_ids[:max_users]

    genre_t=torch.from_numpy(genre_matrix).to(device)

    precisions,recalls,ndcgs=[],[],[]
    items_full=torch.arange(n_movies,dtype=torch.long,device=device)

    t0=time.time()
    for idx,uid in enumerate(user_ids):
        test_items=test_by_user[uid]
        if relevance_threshold is not None:
            relevant={i for i,r in test_items if r>=relevance_threshold}
        else:
            relevant={i for i,_ in test_items}
        if len(relevant)==0:
            continue

        u_tensor=torch.full((n_movies,),uid,dtype=torch.long,device=device)
        scores=model(u_tensor,items_full,genre_t).cpu().numpy()

        seen=train_by_user[uid]
        if seen:
            scores[list(seen)]=-np.inf
            
        if k>=n_movies:
            topk_idx=np.argsort(-scores)[:k]
        else:
            topk_idx=np.argpartition(-scores,k)[:k]
            topk_idx=topk_idx[np.argsort(-scores[topk_idx])]
        recommended=topk_idx.tolist()

        p=precision_at_k(recommended,relevant,k)
        r=recall_at_k(recommended,relevant,k)
        n=ndcg_at_k(recommended,relevant,k)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

        if verbose and (idx+1)%500==0:
            elapsed=time.time()-t0
            rate=(idx+1)/elapsed
            eta=(len(user_ids)-idx-1)/rate
            print(f"  [topN] {idx+1}/{len(user_ids)} users ({rate:.1f}/s, ETA {eta:.0f}s)")

    P=float(np.mean(precisions)) if precisions else 0.0
    R=float(np.mean(recalls)) if recalls else 0.0
    F=f1_from_pr(P,R)
    N=float(np.mean(ndcgs)) if ndcgs else 0.0

    return {
        f"Precision@{k}":P,
        f"Recall@{k}":R,
        f"F1@{k}":F,
        f"NDCG@{k}":N,
        "n_users_evaluated":len(precisions),
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=str,default="data/processed.pkl")
    parser.add_argument("--ckpt",type=str,default="checkpoints/hybrid_ncf.pt")
    parser.add_argument("--k",type=int,default=10)
    parser.add_argument("--max_users",type=int,default=None,
                        help="Evaluate top-N on only the first N users (for speed).")
    parser.add_argument("--relevance_threshold",type=float,default=None,
                        help="If set, only test items with rating >= this count as relevant.")
    parser.add_argument("--out",type=str,default="results/eval_metrics.json")
    args=parser.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    with open(args.data,"rb") as f:
        data=pickle.load(f)

    ckpt=torch.load(args.ckpt,map_location=device,weights_only=False)
    model=HybridNCF(
        n_users=data["n_users"],
        n_movies=data["n_movies"],
        n_genres=data["n_genres"],
        gmf_dim=ckpt["args"].get("gmf_dim",32),
        mlp_dim=ckpt["args"].get("mlp_dim",32),
        dropout=ckpt["args"].get("dropout",0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("[eval] Computing rating metrics...")
    rating_metrics,_=compute_rating_metrics(
        model,data["test"],data["genre_matrix"],device
    )
    print(f"  MAE  = {rating_metrics['MAE']:.4f}")
    print(f"  RMSE = {rating_metrics['RMSE']:.4f}")

    print(f"[eval] Computing Top-{args.k} metrics (this takes a while: one forward pass per user)...")
    topn_metrics=compute_topn_metrics(
        model,
        data["train"],
        data["test"],
        data["genre_matrix"],
        n_movies=data["n_movies"],
        device=device,
        k=args.k,
        max_users=args.max_users,
        relevance_threshold=args.relevance_threshold,
    )
    for key,val in topn_metrics.items():
        if isinstance(val,float):
            print(f"  {key} = {val:.4f}")
        else:
            print(f"  {key} = {val}")

    import json
    all_metrics={**rating_metrics,**topn_metrics}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    with open(args.out,"w") as f:
        json.dump(all_metrics,f,indent=2)
    print(f"[eval] Metrics saved to {args.out}")

if __name__=="__main__":
    main()