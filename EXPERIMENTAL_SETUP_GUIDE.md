# Training Data Guide & Protein Embedding Experiment

## 📊 Datasets Used in Original SurvPath Paper

The original paper used **5 TCGA cancer types** with pre-extracted features:

| Cancer Type | Full Name | Samples | Data Available |
|-------------|-----------|---------|----------------|
| **BLCA** | Bladder Urothelial Carcinoma | 423 | ✓ Metadata, Splits |
| **BRCA** | Breast Invasive Carcinoma | 931 | ✓ Metadata, Splits |
| **COADREAD** | Colon & Rectal Adenocarcinoma | 301 | ✓ Metadata, Splits |
| **HNSC** | Head & Neck Squamous Cell Carcinoma | 414 | ✓ Metadata, Splits |
| **STAD** | Stomach Adenocarcinoma | 343 | ✓ Metadata, Splits |

**Total:** ~2,400 patients across 5 cancer types

---

## 📁 Data Components

For each patient, you need 3 components:

### 1. **Whole Slide Image (WSI) Features**
- **Format:** `.pt` files (PyTorch tensors)
- **Shape:** `[num_patches, 768]` (typically 4096 patches)
- **Extraction:** CTransPath (Swin Transformer pretrained on histology)
- **Location:** NOT included in repo (too large ~100GB+)
- **Download:** Process your own from [TCGA GDC Portal](https://portal.gdc.cancer.gov)

### 2. **RNA-seq (Transcriptomics) Data**
- **Format:** CSV files
- **Location:** `datasets_csv/raw_rna_data/combine/{cancer_type}/`
- **Status:** ✅ **Included in repo**
- **Source:** UCSC Xena database
- **Pathway grouping:** 331 pathways (Reactome + Hallmarks combined)

### 3. **Protein Embeddings** (NEW - Your Addition)
- **Format:** `.pt` files
- **Shape:** `[num_proteins, embedding_dim]` (e.g., `[100, 1280]`)
- **Source:** Need to generate (see below)
- **Status:** ❌ Not yet created

---

## 🎯 Recommended Approach: Small-Scale Validation Experiment

Since full WSI features are large (~100GB+), I recommend starting with a **small subset** to validate the protein integration works and shows improvement.

### **Strategy: Use BRCA subset (20-50 patients)**

**Why BRCA?**
- Largest dataset (931 samples)
- Well-studied cancer with known protein biomarkers (TP53, BRCA1/2, HER2)
- Rich survival data

---

## 🧪 Experimental Setup: 2-Modality vs 3-Modality

### **Experiment Design**

```
Baseline (2-modality):  Pathways + Histology
Enhanced (3-modality):  Pathways + Proteins + Histology

Comparison Metrics:
  - C-index (concordance index)
  - IBS (Integrated Brier Score)
  - iAUC (time-dependent AUC)
```

### **Step-by-Step Guide**

---

## 📋 Step 1: Select a Subset

### Option A: First 50 patients from BRCA

```python
# Create subset split file
import pandas as pd

# Load original split
split = pd.read_csv('splits/5foldcv/tcga_brca/splits_1.csv')

# Take first 25 train, 10 val
subset_split = pd.DataFrame({
    'train': split['train'].iloc[:25],
    'val': split['val'].iloc[:10]
})

# Save
subset_split.to_csv('splits/5foldcv/tcga_brca/splits_subset.csv', index=False)
```

### Option B: Use existing fold 1 (recommended for reproducibility)

Just use `splits_1.csv` as-is for one fold experiment.

---

## 📋 Step 2: Get WSI Features

**Option 1: Download Pre-extracted** (if available from authors)
- Email authors (avaidya@mit.edu) asking for BRCA subset features

**Option 2: Extract Yourself**

```bash
# 1. Download BRCA WSI from TCGA
# https://portal.gdc.cancer.gov/projects/TCGA-BRCA

# 2. Extract features using CLAM + CTransPath
# https://github.com/mahmoodlab/CLAM
# https://github.com/Xiyue-Wang/TransPath

# 3. Save as .pt files: {slide_id}.pt
```

**Option 3: Use Dummy Features for Initial Testing**

```python
# Create dummy WSI features for testing pipeline
import torch
import os

case_ids = ['TCGA-A1-A0SD', 'TCGA-A1-A0SF', ...]  # Your subset

for case_id in case_ids:
    # Generate dummy features
    dummy_wsi = torch.randn(4096, 768)  # 4096 patches, 768-dim

    # Save
    torch.save(dummy_wsi, f'wsi_features/{case_id}.pt')
```

---

## 📋 Step 3: Generate Protein Embeddings

### **3.1: Identify Key Proteins for BRCA**

For breast cancer, focus on well-known prognostic proteins:

```python
# Top 100 cancer-related proteins (examples)
CANCER_PROTEINS = [
    # DNA repair / tumor suppressors
    'TP53', 'BRCA1', 'BRCA2', 'PTEN', 'ATM', 'CHEK2',

    # Cell cycle
    'CDK4', 'CDK6', 'CCND1', 'RB1', 'MYC',

    # RTKs / Growth factors
    'ERBB2', 'EGFR', 'MET', 'FGFR1', 'IGF1R',

    # PI3K/AKT pathway
    'PIK3CA', 'AKT1', 'MTOR', 'TSC1', 'TSC2',

    # Hormone receptors
    'ESR1', 'ESR2', 'PGR', 'AR',

    # ... add 60 more to reach 100
]
```

### **3.2: Get Protein Sequences**

```python
from Bio import SeqIO
from Bio import Entrez
import requests

def get_protein_sequence(gene_name):
    """
    Fetch protein sequence from UniProt
    """
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}+AND+organism_id:9606&format=fasta&limit=1"
    response = requests.get(url)

    if response.ok:
        # Parse FASTA
        lines = response.text.split('\n')
        sequence = ''.join([l for l in lines[1:] if not l.startswith('>')])
        return sequence
    return None

# Example
tp53_seq = get_protein_sequence('TP53')
print(f"TP53 sequence length: {len(tp53_seq)}")
```

### **3.3: Generate ESM-2 Embeddings**

```python
import torch
from transformers import AutoTokenizer, EsmModel

# Load ESM-2
print("Loading ESM-2 model...")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model.eval()

def encode_protein(sequence):
    """Encode a protein sequence with ESM-2"""
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling over sequence length
        embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 1280]

    return embedding.squeeze(0)  # [1280]

# Encode all proteins
protein_embeddings = []
protein_names = []

for gene in CANCER_PROTEINS:
    seq = get_protein_sequence(gene)
    if seq:
        emb = encode_protein(seq)
        protein_embeddings.append(emb)
        protein_names.append(gene)
        print(f"Encoded {gene}: {len(seq)} aa")

# Stack into single tensor
all_embeddings = torch.stack(protein_embeddings)  # [100, 1280]
print(f"Final shape: {all_embeddings.shape}")

# Save for EACH patient (same proteins for all patients)
for case_id in case_ids:
    torch.save({
        'embeddings': all_embeddings,
        'protein_ids': protein_names
    }, f'protein_embeddings/{case_id}_protein.pt')
```

---

## 📋 Step 4: Run Baseline (2-Modality)

```bash
# Train WITHOUT proteins (original SurvPath)
python main.py \
    --study tcga_brca \
    --modality survpath \
    --data_root_dir ./wsi_features \
    --omics_dir datasets_csv/raw_rna_data/combine/brca \
    --label_file datasets_csv/metadata/tcga_brca.csv \
    --split_dir splits/5foldcv/tcga_brca \
    --which_splits 5foldcv \
    --type_of_path combine \
    --k 1 \
    --k_start 0 \
    --k_end 1 \
    --batch_size 1 \
    --lr 0.0001 \
    --opt adam \
    --max_epochs 20 \
    --n_classes 4 \
    --num_patches 4096 \
    --wsi_projection_dim 256 \
    --bag_loss nll_surv \
    --alpha_surv 0.5 \
    --results_dir results_baseline_2mod

# This will output: results_baseline_2mod/summary.csv
# Columns: val_cindex, val_cindex_ipcw, val_IBS, val_iauc
```

---

## 📋 Step 5: Run Enhanced (3-Modality)

```bash
# Train WITH proteins (your enhancement!)
python main.py \
    --study tcga_brca \
    --modality survpath \
    --data_root_dir ./wsi_features \
    --omics_dir datasets_csv/raw_rna_data/combine/brca \
    --protein_dir ./protein_embeddings \
    --num_proteins 100 \
    --protein_embedding_dim 1280 \
    --label_file datasets_csv/metadata/tcga_brca.csv \
    --split_dir splits/5foldcv/tcga_brca \
    --which_splits 5foldcv \
    --type_of_path combine \
    --k 1 \
    --k_start 0 \
    --k_end 1 \
    --batch_size 1 \
    --lr 0.0001 \
    --opt adam \
    --max_epochs 20 \
    --n_classes 4 \
    --num_patches 4096 \
    --wsi_projection_dim 256 \
    --bag_loss nll_surv \
    --alpha_surv 0.5 \
    --results_dir results_enhanced_3mod

# This will output: results_enhanced_3mod/summary.csv
```

---

## 📋 Step 6: Compare Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
baseline = pd.read_csv('results_baseline_2mod/summary.csv')
enhanced = pd.read_csv('results_enhanced_3mod/summary.csv')

# Compare
metrics = ['val_cindex', 'val_cindex_ipcw', 'val_IBS', 'val_iauc']

print("="*60)
print("COMPARISON: 2-Modality vs 3-Modality")
print("="*60)

for metric in metrics:
    base_val = baseline[metric].values[0]
    enh_val = enhanced[metric].values[0]

    improvement = ((enh_val - base_val) / base_val) * 100

    print(f"\n{metric}:")
    print(f"  Baseline (2-mod):  {base_val:.4f}")
    print(f"  Enhanced (3-mod):  {enh_val:.4f}")
    print(f"  Improvement:       {improvement:+.2f}%")

# Visualize
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

for i, metric in enumerate(metrics):
    ax[i].bar(['Baseline\n(2-mod)', 'Enhanced\n(3-mod)'],
              [baseline[metric].values[0], enhanced[metric].values[0]])
    ax[i].set_title(metric)
    ax[i].set_ylabel('Score')

plt.tight_layout()
plt.savefig('protein_improvement.png', dpi=300)
print("\nPlot saved to protein_improvement.png")
```

---

## 📊 Expected Results

Based on similar multimodal fusion studies, you might see:

### **Conservative Estimates**
```
C-index improvement:     +1-3%
IBS improvement:         +2-5%
iAUC improvement:        +1-4%
```

### **If Proteins are Highly Prognostic**
```
C-index improvement:     +3-8%
IBS improvement:         +5-10%
iAUC improvement:        +4-8%
```

**Why might it improve?**
- Proteins capture functional state (post-translational)
- Complement genomic data (RNA → protein not 1:1)
- Protein-histology interactions reveal phenotypes

---

## 🚀 Quick Start: Minimal Test with Dummy Data

If you just want to **verify the pipeline works**:

```bash
# 1. Create dummy test script
cat > test_protein_pipeline.py << 'EOF'
import torch
import os
import pandas as pd

# Create minimal test data
os.makedirs('test_data/wsi_features', exist_ok=True)
os.makedirs('test_data/protein_embeddings', exist_ok=True)

# Get first 5 patients from BRCA
metadata = pd.read_csv('datasets_csv/metadata/tcga_brca.csv')
test_cases = metadata['case_id'].unique()[:5]

print(f"Creating dummy data for {len(test_cases)} patients...")

for case_id in test_cases:
    # Dummy WSI features
    torch.save(torch.randn(4096, 768), f'test_data/wsi_features/{case_id}.pt')

    # Dummy protein embeddings
    torch.save({
        'embeddings': torch.randn(100, 1280),
        'protein_ids': [f'PROT{i}' for i in range(100)]
    }, f'test_data/protein_embeddings/{case_id}_protein.pt')

print("Dummy data created!")

# Create tiny split
tiny_split = pd.DataFrame({
    'train': test_cases[:3],
    'val': test_cases[3:]
})
os.makedirs('test_data/splits', exist_ok=True)
tiny_split.to_csv('test_data/splits/splits_1.csv', index=False)

print("Ready to test!")
EOF

python test_protein_pipeline.py

# 2. Test 2-modality
python main.py \
    --study tcga_brca \
    --modality survpath \
    --data_root_dir test_data/wsi_features \
    --omics_dir datasets_csv/raw_rna_data/combine/brca \
    --label_file datasets_csv/metadata/tcga_brca.csv \
    --split_dir test_data/splits \
    --type_of_path combine \
    --k 1 --max_epochs 2 --batch_size 1 \
    --results_dir test_results_2mod

# 3. Test 3-modality (with proteins)
python main.py \
    --study tcga_brca \
    --modality survpath \
    --data_root_dir test_data/wsi_features \
    --omics_dir datasets_csv/raw_rna_data/combine/brca \
    --protein_dir test_data/protein_embeddings \
    --num_proteins 100 \
    --protein_embedding_dim 1280 \
    --label_file datasets_csv/metadata/tcga_brca.csv \
    --split_dir test_data/splits \
    --type_of_path combine \
    --k 1 --max_epochs 2 --batch_size 1 \
    --results_dir test_results_3mod

echo "Pipeline test complete! Check test_results_*/"
```

---

## 📚 Resources

### **Data Sources**
- **TCGA Portal:** https://portal.gdc.cancer.gov
- **UCSC Xena:** https://xenabrowser.net
- **UniProt (proteins):** https://www.uniprot.org
- **MSigDB (pathways):** https://www.gsea-msigdb.org

### **Tools**
- **CTransPath (WSI features):** https://github.com/Xiyue-Wang/TransPath
- **CLAM (WSI processing):** https://github.com/mahmoodlab/CLAM
- **ESM-2 (protein embeddings):** https://github.com/facebookresearch/esm

---

## ⚠️ Important Notes

1. **WSI Features are Large:**
   - Full BRCA: ~100GB
   - Consider starting with 20-50 patients (~2-5GB)

2. **Protein Embeddings:**
   - ESM-2 model: ~2.5GB download
   - Encoding 100 proteins: ~5 minutes

3. **Training Time:**
   - 20 epochs, 50 patients: ~30-60 minutes (GPU)
   - Full BRCA (931 samples): ~3-5 hours (GPU)

4. **Baseline First:**
   - Always run 2-modality baseline first
   - This validates your setup works
   - Then add proteins to see delta

---

## 🎯 Recommended Path

**For Quick Validation:**
1. Use dummy WSI features (5-10 patients)
2. Generate real protein embeddings (shows concept)
3. Run 2 epochs to verify pipeline works
4. Check shapes, no crashes = success!

**For Real Comparison:**
1. Get subset of real WSI features (50-100 patients)
2. Generate protein embeddings with ESM-2
3. Train baseline (2-mod) for 20 epochs
4. Train enhanced (3-mod) for 20 epochs
5. Compare metrics

**For Publication:**
1. Use full datasets (all 5 cancer types)
2. 5-fold cross-validation
3. Multiple runs with different seeds
4. Statistical significance testing

---

**Status:** Ready to run experiments!

See `BUG_FIX_REPORT.md` - All data flow issues are fixed.
