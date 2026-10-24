# CS 550 Project Makefile
# Convenience wrappers around the pipeline steps.
#
# Typical usage on Amarel (inside the cs550 conda env):
#     make smoke          # 1-minute synthetic sanity check
#     make data           # download + preprocess MovieLens 1M
#     make train          # train Hybrid-NCF
#     make eval           # compute all 6 metrics
#     make baselines      # compute baseline numbers
#     make fill           # fill placeholders in report + slides
#     make all            # data -> train -> eval -> baselines -> fill
#     make demo           # launch the Streamlit demo (local)

PY ?= python

.PHONY: all smoke data train eval baselines fill demo clean

all: data train eval baselines fill

smoke:
	$(PY) notebooks/smoke_test.py

data:
	bash scripts/download_movielens.sh
	$(PY) data/preprocess.py --data_dir ./ml-1m --out_dir ./data

train:
	$(PY) train.py \
	    --data data/processed.pkl \
	    --epochs 25 --batch_size 2048 \
	    --lr 1e-3 --weight_decay 1e-5 \
	    --gmf_dim 32 --mlp_dim 32 --dropout 0.2 \
	    --patience 4 \
	    --ckpt checkpoints/hybrid_ncf.pt \
	    --curve_csv results/training_curve.csv

eval:
	$(PY) -m evaluation.evaluate \
	    --data data/processed.pkl \
	    --ckpt checkpoints/hybrid_ncf.pt \
	    --k 10 \
	    --out results/eval_metrics.json

baselines:
	$(PY) scripts/baselines.py \
	    --data data/processed.pkl \
	    --k 10 \
	    --out results/baseline_metrics.json

fill:
	$(PY) scripts/fill_report.py \
	    --template report/report.tex \
	    --metrics  results/eval_metrics.json \
	    --curve_csv results/training_curve.csv \
	    --out      report/report_filled.tex
	@echo "---"
	@echo "Rebuilding slides with real numbers..."
	node slides/build_slides.js results/eval_metrics.json

demo:
	streamlit run demo/app.py -- \
	    --data data/processed.pkl \
	    --ckpt checkpoints/hybrid_ncf.pt \
	    --metrics_json results/eval_metrics.json

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -f results/eval_metrics.json results/baseline_metrics.json results/training_curve.csv
	rm -f report/report_filled.tex
	@echo "Kept: data/processed.pkl, checkpoints/, ml-1m/  (run 'make distclean' to remove these too)"

distclean: clean
	rm -rf ml-1m data/processed.pkl checkpoints/* results/*
	touch checkpoints/.gitkeep results/.gitkeep
