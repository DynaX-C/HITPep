# HITPep

HITPep is a hierarchical interaction-learning framework for protein–peptide complex quality assessment and model ranking.

Instead of representing complex quality with a single prediction target, HITPep decomposes structural quality across **atomic, residue, and global scales**. Local protein–peptide structures are encoded through complementary non-bonded, bonded, bond-angle, and dihedral interaction topologies, and the learned representations are progressively propagated from atoms to residues and the whole complex.

HITPep predicts five structural quality components:

- **Atom Score**
- **Residue Geometry Score**
- **Residue Interaction Score**
- **Global Geometry Score**
- **Global Interaction Score**

These components provide structurally resolved assessments of candidate complexes and are further integrated into a unified **HITPep Score** for model ranking.

---

## Installation

Create the Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate HITPep
```

---

## Inference

HITPep supports two common input formats:

1. receptor and peptide structures provided separately;
2. receptor–peptide complex structures provided as complete complexes.

The preprocessing step constructs the hierarchical interaction graphs and generates the features required for HITPep inference.

### 1. Receptor and peptide provided separately

Example:

```bash
python data_pre_workflow.py \
    --decoy_protein_pdb example/1B9J_HPEPDOCK/receptor.pdb \
    --decoy_peptide_pdb example/1B9J_HPEPDOCK/decoys_HPEPDOCK.pdb \
    --save_graph_pt example/1B9J_HPEPDOCK/graph.pt \
    --out_dir example/1B9J_HPEPDOCK/cache \
    --esm_model /path/to/esm2_model/
```

Run HITPep inference:

```bash
python inference.py \
    --graph_pt example/1B9J_HPEPDOCK/graph.pt \
    --ckpt checkpoints/best_model_hitpep.pt \
    --batch_size 8 \
    --save_pred_csv 1B9J_HPEPDOCK.csv
```

Here:

- `--decoy_protein_pdb` specifies the receptor structure;
- `--decoy_peptide_pdb` specifies the peptide decoy structure or multi-model PDB file;
- `--save_graph_pt` specifies the output graph file;
- `--out_dir` stores intermediate preprocessing files;
- `--esm_model` specifies the local ESM-2 model directory.

---

### 2. Receptor and peptide provided as a complex

For complex structures, specify the receptor and peptide chain IDs.

Example:

```bash
python data_pre_workflow.py \
    --complex_pdb "example/1AWR_AFM/models_AFM/unrelaxed_*.pdb" \
    --protein_chain B \
    --peptide_chain C \
    --save_graph_pt example/1AWR_AFM/graph.pt \
    --out_dir example/1AWR_AFM/cache \
    --esm_model /path/to/esm2_model/
```

Run HITPep inference:

```bash
python inference.py \
    --graph_pt example/1AWR_AFM/graph.pt \
    --ckpt checkpoints/best_model_hitpep.pt \
    --batch_size 8 \
    --save_pred_csv 1AWR_AFM.csv
```

Here:

- `--complex_pdb` can specify a single complex or multiple structures using wildcards;
- `--protein_chain` specifies the receptor chain;
- `--peptide_chain` specifies the peptide chain.

The output CSV contains the predicted structural quality components and the final HITPep ranking score for each candidate structure.

---

## Training HITPep on a Custom Dataset

The training dataset used in this work is derived from the protein–peptide decoy dataset released with GraphPep:

https://zenodo.org/records/17097750

Training HITPep on a custom dataset consists of four main preprocessing steps:

1. structure preprocessing;
2. ESM-2 feature generation;
3. scale parameter calculation from the training set;
4. graph dataset construction.

---

### Data preprocessing

#### Training set

##### Step 1. Structure preprocessing

```bash
python -m data.data_preprocess \
    --csv_path data/Training_set/split_data/train.csv \
    --work_dir data/Training_set \
    --save_path systems_train.pt \
    --num_workers 8
```

##### Step 2. Generate ESM-2 residue embeddings

```bash
python -m data.plm_features \
    --systems_path systems_train.pt \
    --esm_model /path/to/esm2_model/ \
    --overwrite
```

##### Step 3. Calculate training-set scale parameters

The scale parameters used for structural quality labels are determined from the training set using the specified percentile.

```bash
python -m data.tau_cal \
    --systems_path systems_train.pt \
    --csv_path data/Training_set/split_data/train.csv \
    --save_path tau.json \
    --percentile 75
```

##### Step 4. Construct the training dataset

```bash
python -m data.dataset \
    --systems_path systems_train.pt \
    --csv_path data/Training_set/split_data/train.csv \
    --tau_path tau.json \
    --save_path train.pt \
    --num_workers 16
```

---

#### Validation set

Process the validation structures:

```bash
python -m data.data_preprocess \
    --csv_path data/Training_set/split_data/valid.csv \
    --work_dir data/Training_set \
    --save_path systems_valid.pt \
    --num_workers 8
```

Generate ESM-2 embeddings:

```bash
python -m data.plm_features \
    --systems_path systems_valid.pt \
    --esm_model /path/to/esm2_model/ \
    --overwrite
```

Construct the validation dataset using the **same `tau.json` calculated from the training set**:

```bash
python -m data.dataset \
    --systems_path systems_valid.pt \
    --csv_path data/Training_set/split_data/valid.csv \
    --tau_path tau.json \
    --save_path valid.pt \
    --num_workers 16
```

---

### Training

HITPep is trained using **Multi-Scale Curriculum Optimization (MSCO)**. Training progressively changes the relative importance of atomic-, residue-, and global-scale objectives while keeping all five prediction tasks active.

The three stages emphasize:

- **Stage I:** atomic-scale quality;
- **Stage II:** residue-scale quality;
- **Stage III:** global-scale quality.

Run training with:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --train_pt train.pt \
    --val_pt valid.pt \
    --save_dir runs/hitpep \
    \
    --node_features_dim 38 \
    --hidden_channels 128 \
    --edge_dim 8 \
    --num_layers_nba 2 \
    --num_layers_bda 2 \
    --num_layers_bb 2 \
    --num_layers_aa 2 \
    \
    --gat_heads 4 \
    --gat_concat \
    --gat_negative_slope 0.1 \
    --gat_dropout 0.1 \
    --gat_bias \
    --gat_residual \
    \
    --dropout 0.1 \
    --dist_cutoff 6.0 \
    --residue_edge_dim 8 \
    --use_esm \
    --esm_dim 1280 \
    --use_orig_emb \
    --use_hit \
    \
    --batch_size 128 \
    --num_workers 2 \
    --weight_decay 1e-5 \
    --seed 42 \
    \
    --epochs 500 \
    --transition_epochs 0 \
    --transition_mode cosine \
    --patience_stage1 10 \
    --patience_stage2 20 \
    --patience_stage3 30 \
    \
    --lr_stage1 5e-4 \
    --lr_stage2 4e-4 \
    --lr_stage3 3e-4 \
    \
    --stage1_w_atom 1.0 \
    --stage1_w_res_geom 0.3 \
    --stage1_w_res_int 0.3 \
    --stage1_w_glb_geom 0.1 \
    --stage1_w_glb_int 0.1 \
    \
    --stage2_w_atom 0.1 \
    --stage2_w_res_geom 1.0 \
    --stage2_w_res_int 1.0 \
    --stage2_w_glb_geom 0.3 \
    --stage2_w_glb_int 0.3 \
    \
    --stage3_w_atom 0.1 \
    --stage3_w_res_geom 0.3 \
    --stage3_w_res_int 0.3 \
    --stage3_w_glb_geom 1.0 \
    --stage3_w_glb_int 1.0
```

The trained checkpoints and training logs are saved to:

```text
runs/hitpep/
```

---

### Pretrained Model

The pretrained HITPep model used in our study is provided at:

```text
checkpoints/best_model_hitpep.pt
```

---