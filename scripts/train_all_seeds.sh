#!/bin/bash
# train_all_seeds.sh

echo "Training baselines (XGBoost, MLP, GAT, RGCN)..."
python3 scripts/baselines.py --model all --seed 42
cp results/xgboost_predictions.npz results/42_xgboost_predictions.npz
cp results/mlp_predictions.npz results/42_mlp_predictions.npz
cp results/gat_predictions.npz results/42_gat_predictions.npz
cp results/rgcn_predictions.npz results/42_rgcn_predictions.npz

python3 scripts/baselines.py --model all --seed 43
cp results/xgboost_predictions.npz results/43_xgboost_predictions.npz
cp results/mlp_predictions.npz results/43_mlp_predictions.npz
cp results/gat_predictions.npz results/43_gat_predictions.npz
cp results/rgcn_predictions.npz results/43_rgcn_predictions.npz

python3 scripts/baselines.py --model all --seed 44
cp results/xgboost_predictions.npz results/44_xgboost_predictions.npz
cp results/mlp_predictions.npz results/44_mlp_predictions.npz
cp results/gat_predictions.npz results/44_gat_predictions.npz
cp results/rgcn_predictions.npz results/44_rgcn_predictions.npz

echo "Training THA-GAT across seeds..."
python3 scripts/train_thagat.py --baseline --seed 42
cp results/thagat_predictions.npz results/42_thagat_predictions.npz
cp results/graphsage_predictions.npz results/42_graphsage_predictions.npz

python3 scripts/train_thagat.py --baseline --seed 43
cp results/thagat_predictions.npz results/43_thagat_predictions.npz
cp results/graphsage_predictions.npz results/43_graphsage_predictions.npz

python3 scripts/train_thagat.py --baseline --seed 44
cp results/thagat_predictions.npz results/44_thagat_predictions.npz
cp results/graphsage_predictions.npz results/44_graphsage_predictions.npz

echo "Training Ablation Variants..."
python3 scripts/ablation_study.py

echo "Running Threshold Optimization..."
python3 scripts/optimize_threshold.py --model thagat --seed 42
python3 scripts/optimize_threshold.py --model xgboost --seed 42
python3 scripts/optimize_threshold.py --model ensemble --seed 42

echo "All multi-seed experiments complete!"
