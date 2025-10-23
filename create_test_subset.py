"""
Create a small test subset for validating protein embedding integration

This script creates:
1. Dummy WSI features for a subset of BRCA patients
2. Real protein embeddings using ESM-2 (optional)
3. Subset split files
4. Directory structure

Usage:
  python create_test_subset.py --num_patients 10 --use_real_proteins
"""

import argparse
import os
import torch
import pandas as pd
from pathlib import Path

def create_dummy_wsi_features(case_ids, output_dir, num_patches=4096, feat_dim=768):
    """Create dummy WSI features for testing"""
    print(f"\nCreating dummy WSI features...")
    os.makedirs(output_dir, exist_ok=True)

    for case_id in case_ids:
        # Generate random features
        features = torch.randn(num_patches, feat_dim)

        # Save
        output_path = os.path.join(output_dir, f'{case_id}.pt')
        torch.save(features, output_path)

    print(f"✓ Created {len(case_ids)} dummy WSI feature files in {output_dir}")

def create_dummy_protein_embeddings(case_ids, output_dir, num_proteins=100, emb_dim=1280):
    """Create dummy protein embeddings for testing"""
    print(f"\nCreating dummy protein embeddings...")
    os.makedirs(output_dir, exist_ok=True)

    # Same embeddings for all patients (reasonable for test)
    protein_embeddings = torch.randn(num_proteins, emb_dim)
    protein_ids = [f'PROTEIN_{i:03d}' for i in range(num_proteins)]

    for case_id in case_ids:
        data = {
            'embeddings': protein_embeddings,
            'protein_ids': protein_ids
        }

        output_path = os.path.join(output_dir, f'{case_id}_protein.pt')
        torch.save(data, output_path)

    print(f"✓ Created {len(case_ids)} dummy protein embedding files in {output_dir}")

def create_real_protein_embeddings(case_ids, output_dir, num_proteins=100):
    """Create real protein embeddings using ESM-2"""
    print(f"\nCreating REAL protein embeddings using ESM-2...")
    print("This will download ESM-2 model (~2.5GB) if not already cached...")

    try:
        from transformers import AutoTokenizer, EsmModel
        import requests
    except ImportError:
        print("ERROR: Need transformers and requests libraries")
        print("Install with: pip install transformers requests")
        return False

    # Load ESM-2
    print("Loading ESM-2 model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model.eval()

    # Top cancer-related proteins
    CANCER_PROTEINS = [
        # DNA repair / tumor suppressors
        'TP53', 'BRCA1', 'BRCA2', 'PTEN', 'ATM', 'CHEK2', 'RB1', 'APC',
        # Cell cycle
        'CDK4', 'CDK6', 'CCND1', 'CCNE1', 'MYC', 'E2F1', 'CDKN2A', 'CDKN1A',
        # RTKs / Growth factors
        'ERBB2', 'EGFR', 'MET', 'FGFR1', 'FGFR2', 'IGF1R', 'PDGFRA', 'KIT',
        # PI3K/AKT/mTOR pathway
        'PIK3CA', 'AKT1', 'AKT2', 'MTOR', 'TSC1', 'TSC2', 'PTEN', 'INPP4B',
        # RAS/RAF/MEK pathway
        'KRAS', 'NRAS', 'BRAF', 'MAP2K1', 'MAP2K2', 'RAF1',
        # Hormone receptors
        'ESR1', 'ESR2', 'PGR', 'AR',
        # Apoptosis
        'BCL2', 'BAX', 'CASP3', 'CASP9', 'BID',
        # Cell adhesion
        'CDH1', 'CTNNB1', 'CTNNA1', 'CTNND1',
        # Transcription factors
        'TP63', 'TP73', 'FOXO3', 'STAT3', 'JUN', 'FOS',
        # Chromatin remodeling
        'ARID1A', 'SMARCA4', 'KMT2C', 'KMT2D',
        # DNA methylation
        'DNMT1', 'DNMT3A', 'DNMT3B', 'TET2',
        # Metabolism
        'IDH1', 'IDH2', 'PKM', 'LDHA', 'HK2',
        # Immune checkpoints
        'PDCD1', 'CD274', 'CTLA4', 'LAG3',
        # Angiogenesis
        'VEGFA', 'VEGFR2', 'HIF1A', 'ANGPT1',
        # Others
        'MDM2', 'MDM4', 'NOTCH1', 'JAK2', 'GATA3', 'FOXA1',
        'NF1', 'SMAD4', 'FBXW7', 'FAT1', 'MAP3K1', 'NCOR1',
        'RUNX1', 'CBFB', 'TBX3', 'PIK3R1', 'ERBB3', 'SF3B1'
    ][:num_proteins]

    def get_protein_sequence(gene_name):
        """Fetch protein sequence from UniProt"""
        url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}+AND+organism_id:9606&format=fasta&limit=1"
        try:
            response = requests.get(url, timeout=10)
            if response.ok and response.text:
                lines = response.text.split('\n')
                sequence = ''.join([l.strip() for l in lines[1:] if l and not l.startswith('>')])
                return sequence
        except:
            pass
        return None

    def encode_protein(sequence):
        """Encode protein with ESM-2"""
        inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding

    # Encode all proteins
    protein_embeddings = []
    protein_names = []

    print(f"Encoding {len(CANCER_PROTEINS)} cancer-related proteins...")
    for i, gene in enumerate(CANCER_PROTEINS):
        print(f"  [{i+1}/{len(CANCER_PROTEINS)}] {gene}...", end=' ')

        seq = get_protein_sequence(gene)
        if seq and len(seq) > 0:
            emb = encode_protein(seq)
            protein_embeddings.append(emb)
            protein_names.append(gene)
            print(f"✓ ({len(seq)} aa)")
        else:
            # Fallback to random if sequence not found
            protein_embeddings.append(torch.randn(1280))
            protein_names.append(gene)
            print("✗ (using random)")

    # Stack
    all_embeddings = torch.stack(protein_embeddings)
    print(f"\n✓ Encoded {len(protein_names)} proteins, shape: {all_embeddings.shape}")

    # Save for all cases
    os.makedirs(output_dir, exist_ok=True)
    for case_id in case_ids:
        data = {
            'embeddings': all_embeddings,
            'protein_ids': protein_names
        }
        output_path = os.path.join(output_dir, f'{case_id}_protein.pt')
        torch.save(data, output_path)

    print(f"✓ Saved protein embeddings for {len(case_ids)} patients in {output_dir}")
    return True

def create_subset_splits(case_ids, output_dir, train_frac=0.7):
    """Create train/val split files"""
    print(f"\nCreating subset split files...")
    os.makedirs(output_dir, exist_ok=True)

    # Split into train/val
    n_train = int(len(case_ids) * train_frac)
    train_cases = case_ids[:n_train]
    val_cases = case_ids[n_train:]

    # Create split file
    split_df = pd.DataFrame({
        'train': list(train_cases) + [''] * (len(val_cases) - len(train_cases)) if len(val_cases) > len(train_cases) else list(train_cases),
        'val': list(val_cases) + [''] * (len(train_cases) - len(val_cases)) if len(train_cases) > len(val_cases) else list(val_cases)
    })

    # Pad to same length
    max_len = max(len(train_cases), len(val_cases))
    split_df = pd.DataFrame({
        'train': list(train_cases) + [''] * (max_len - len(train_cases)),
        'val': list(val_cases) + [''] * (max_len - len(val_cases))
    })

    output_path = os.path.join(output_dir, 'splits_1.csv')
    split_df.to_csv(output_path, index=False)

    print(f"✓ Created split: {len(train_cases)} train, {len(val_cases)} val")
    print(f"  Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create test subset for protein embedding validation')
    parser.add_argument('--num_patients', type=int, default=20,
                        help='Number of patients for test subset (default: 20)')
    parser.add_argument('--dataset', type=str, default='brca', choices=['brca', 'blca', 'coadread', 'hnsc', 'stad'],
                        help='Which TCGA dataset to use (default: brca)')
    parser.add_argument('--use_real_proteins', action='store_true',
                        help='Generate real protein embeddings with ESM-2 (requires transformers library)')
    parser.add_argument('--num_proteins', type=int, default=100,
                        help='Number of proteins to encode (default: 100)')
    parser.add_argument('--output_dir', type=str, default='test_subset',
                        help='Output directory for test data (default: test_subset)')

    args = parser.parse_args()

    print("="*80)
    print("Creating Test Subset for Protein Embedding Validation")
    print("="*80)

    # Load metadata
    metadata_path = f'datasets_csv/metadata/tcga_{args.dataset}.csv'
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file not found: {metadata_path}")
        return

    metadata = pd.read_csv(metadata_path)

    # Get unique case IDs
    all_cases = metadata['case_id'].unique()
    print(f"\nDataset: TCGA-{args.dataset.upper()}")
    print(f"Total available cases: {len(all_cases)}")

    # Select subset
    num_patients = min(args.num_patients, len(all_cases))
    subset_cases = all_cases[:num_patients]

    print(f"Selected subset: {num_patients} patients")
    print(f"Case IDs: {', '.join(subset_cases[:5])}{'...' if num_patients > 5 else ''}")

    # Create directory structure
    base_dir = args.output_dir
    wsi_dir = os.path.join(base_dir, 'wsi_features')
    protein_dir = os.path.join(base_dir, 'protein_embeddings')
    split_dir = os.path.join(base_dir, 'splits')

    # Create WSI features (always dummy)
    create_dummy_wsi_features(subset_cases, wsi_dir)

    # Create protein embeddings
    if args.use_real_proteins:
        success = create_real_protein_embeddings(subset_cases, protein_dir, args.num_proteins)
        if not success:
            print("\nFalling back to dummy protein embeddings...")
            create_dummy_protein_embeddings(subset_cases, protein_dir, args.num_proteins)
    else:
        create_dummy_protein_embeddings(subset_cases, protein_dir, args.num_proteins)

    # Create splits
    create_subset_splits(subset_cases, split_dir)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Created test subset in: {base_dir}/")
    print(f"  - WSI features:       {wsi_dir}/ ({num_patients} files)")
    print(f"  - Protein embeddings: {protein_dir}/ ({num_patients} files)")
    print(f"  - Train/val split:    {split_dir}/splits_1.csv")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Test baseline (2-modality):")
    print(f"""
python main.py \\
    --study tcga_{args.dataset} \\
    --modality survpath \\
    --data_root_dir {wsi_dir} \\
    --omics_dir datasets_csv/raw_rna_data/combine/{args.dataset} \\
    --label_file {metadata_path} \\
    --split_dir {split_dir} \\
    --type_of_path combine \\
    --k 1 --max_epochs 5 --batch_size 1 \\
    --results_dir results_baseline_2mod
""")

    print("\n2. Test enhanced (3-modality with proteins):")
    print(f"""
python main.py \\
    --study tcga_{args.dataset} \\
    --modality survpath \\
    --data_root_dir {wsi_dir} \\
    --omics_dir datasets_csv/raw_rna_data/combine/{args.dataset} \\
    --protein_dir {protein_dir} \\
    --num_proteins {args.num_proteins} \\
    --protein_embedding_dim 1280 \\
    --label_file {metadata_path} \\
    --split_dir {split_dir} \\
    --type_of_path combine \\
    --k 1 --max_epochs 5 --batch_size 1 \\
    --results_dir results_enhanced_3mod
""")

    print("\n3. Compare results:")
    print("""
python -c "
import pandas as pd
baseline = pd.read_csv('results_baseline_2mod/summary.csv')
enhanced = pd.read_csv('results_enhanced_3mod/summary.csv')
print('\\nBaseline (2-mod):', baseline['val_cindex'].values[0])
print('Enhanced (3-mod):', enhanced['val_cindex'].values[0])
print('Improvement:', enhanced['val_cindex'].values[0] - baseline['val_cindex'].values[0])
"
""")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
