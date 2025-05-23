import os
import sys
import numpy as np
import pandas as pd
import torch
import pyBigWig
import tabix
from torch import nn
from torch_fftconv import FFTConv1d
from selene_sdk.targets import Target
import selene_sdk
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load model architecture
class EnhancedSimpleNet(nn.Module):
    def __init__(self, n_motifs):
        super(EnhancedSimpleNet, self).__init__()
        self.conv = nn.Conv1d(4, n_motifs, kernel_size=51, padding=25)
        self.bn_conv = nn.BatchNorm1d(n_motifs * 2)
        self.conv_inr = nn.Conv1d(4, 10, kernel_size=15, padding=7)
        self.bn_conv_inr = nn.BatchNorm1d(20)
        self.conv_sim = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.bn_conv_sim = nn.BatchNorm1d(64)
        self.activation = nn.ReLU()
        self.deconv = FFTConv1d(n_motifs * 2, 10, kernel_size=601, padding=300)
        self.deconv_inr = nn.ConvTranspose1d(20, 10, kernel_size=15, padding=7)
        self.deconv_sim = FFTConv1d(64, 10, kernel_size=601, padding=300)
        self.softplus = nn.Softplus()
        self.scaler = nn.Parameter(torch.ones(1))
        self.scaler2 = nn.Parameter(torch.ones(1))

    def forward(self, x, withscaler=True):
        y = torch.cat([self.conv(x), self.conv(x.flip([1, 2])).flip([2])], 1)
        y = self.bn_conv(y)
        y = self.activation(y) * y
        
        y_inr = torch.cat([self.conv_inr(x), self.conv_inr(x.flip([1, 2])).flip([2])], 1)
        y_inr = self.bn_conv_inr(y_inr)
        y_inr = self.activation(y_inr) * y_inr
        
        y_sim = torch.cat([self.conv_sim(x), self.conv_sim(x.flip([1, 2])).flip([2])], 1)
        y_sim = self.bn_conv_sim(y_sim)
        y_sim = self.activation(y_sim) * y_sim
        
        if withscaler:
            y = self.softplus(y * self.scaler**2)
            y_inr = self.softplus(y_inr)
            y_sim = self.softplus(y_sim * self.scaler2**2)
        else:
            y = self.softplus(y)
            y_inr = self.softplus(y_inr)
            y_sim = self.softplus(y_sim)
            
        y_pred = self.softplus(
            self.deconv(y) + self.deconv_inr(y_inr) + self.deconv_sim(y_sim)
        )
        return y_pred

# Load genomic features class
class GenomicSignalFeatures(Target):
    def __init__(self, input_paths, features, shape, blacklists=None, 
                 blacklists_indices=None, replacement_indices=None, 
                 replacement_scaling_factors=None):
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors
        self.n_features = len(features)
        self.feature_index_dict = dict([(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist) for blacklist in self.blacklists]
            self.initialized = True
            
        if feature_indices is None:
            feature_indices = np.arange(len(self.data))
            
        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise

        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s) - start, 0):int(e) - start] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s) - start, 0):int(e) - start] = 0
            else:
                for (blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor) in zip(
                    self.blacklists, self.blacklists_indices, self.replacement_indices, self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0):int(e) - start] = (
                            wigmat[replacement_indices, np.fmax(int(s) - start, 0):int(e) - start] * replacement_scaling_factor)
        
        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat

def load_model_and_data(model_path, n_motifs, test_chromosomes=["chr8", "chr9"]):
    # Load TSS data
    tsses = pd.read_table(
        "../resources/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v5.highconf.tsv",
        sep="\t",
    )
    
    # Initialize genome and features
    genome = selene_sdk.sequences.Genome(
        input_path="../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
    )
    
    tfeature = GenomicSignalFeatures(
        [
            "../resources/agg.plus.bw.bedgraph.bw",
            "../resources/agg.encodecage.plus.v2.bedgraph.bw",
            "../resources/agg.encoderampage.plus.v2.bedgraph.bw",
            "../resources/agg.plus.grocap.bedgraph.sorted.merged.bw",
            "../resources/agg.plus.allprocap.bedgraph.sorted.merged.bw",
            "../resources/agg.minus.allprocap.bedgraph.sorted.merged.bw",
            "../resources/agg.minus.grocap.bedgraph.sorted.merged.bw",
            "../resources/agg.encoderampage.minus.v2.bedgraph.bw",
            "../resources/agg.encodecage.minus.v2.bedgraph.bw",
            "../resources/agg.minus.bw.bedgraph.bw",
        ],
        [
            "cage_plus", "encodecage_plus", "encoderampage_plus", "grocap_plus", "procap_plus",
            "procap_minus", "grocap_minus", "encoderampage_minus", "encodecage_minus", "cage_minus",
        ],
        (4000,),
        [
            "../resources/fantom.blacklist8.plus.bed.gz",
            "../resources/fantom.blacklist8.minus.bed.gz",
        ],
        [0, 9],
        [1, 8],
        [0.61357, 0.61357],
    )
    
    # Prepare test data
    window_size = 4650
    test_indices = tsses["chr"].isin(test_chromosomes)
    test_tsses = tsses[test_indices].reset_index(drop=True)
    
    seqs = []
    tars = []
    for randi in range(len(test_tsses)):
        row = test_tsses.iloc[randi]
        chrm, pos, strand = row["chr"], row["TSS"], row["strand"]
        offset = 1 if strand == "-" else 0
        
        try:
            seq = genome.get_encoding_from_coords(
                chrm,
                pos - window_size // 2 + offset,
                pos + window_size // 2 + offset,
                strand,
            )
            tar = tfeature.get_feature_data(
                chrm,
                pos - window_size // 2 + offset,
                pos + window_size // 2 + offset,
            )
            if strand == "-":
                tar = tar[::-1, ::-1]

            seqs.append(seq)
            tars.append(tar)
        except Exception as e:
            print(f"Error processing {chrm}:{pos}: {str(e)}")
            continue
    
    if len(seqs) == 0:
        raise ValueError("No valid test sequences found!")
    
    seqs = np.dstack(seqs)
    tars = np.dstack(tars)
    seqs = seqs.transpose([2, 1, 0])
    tars = tars.transpose([2, 0, 1])
    
    # Load model
    model = EnhancedSimpleNet(n_motifs)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    
    return model, seqs, tars, test_tsses

def evaluate_model(model, seqs, tars, batch_size=64):
    predictions = []
    targets = []
    
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i+batch_size]
            batch_tars = tars[i:i+batch_size]
            
            seq_tensor = torch.FloatTensor(batch_seqs).cuda()
            pred = model(seq_tensor, withscaler=False)
            
            predictions.append(pred.cpu().numpy())
            targets.append(batch_tars)
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    return predictions, targets

def calculate_metrics(predictions, targets, feature_names):
    metrics = {}
    eval_window = slice(325, -325)  # Same as used in training
    
    for i, name in enumerate(feature_names):
        pred_flat = predictions[:, i, eval_window].flatten()
        target_flat = targets[:, i, eval_window].flatten()
        
        # Filter zeros to avoid skewing metrics
        non_zero_mask = (target_flat > 0) & (pred_flat > 0)
        pred_filtered = pred_flat[non_zero_mask]
        target_filtered = target_flat[non_zero_mask]
        
        if len(pred_filtered) == 0:
            metrics[name] = {
                'pearson_r': np.nan,
                'spearman_r': np.nan,
                'r2': np.nan,
                'mse': np.nan,
                'poisson_nll': np.nan
            }
            continue
        
        # Pearson correlation
        pearson_r = np.corrcoef(pred_filtered, target_filtered)[0, 1]
        
        # Spearman correlation
        from scipy.stats import spearmanr
        spearman_r = spearmanr(pred_filtered, target_filtered).correlation
        
        # R-squared
        r2 = r2_score(target_filtered, pred_filtered)
        
        # MSE
        mse = np.mean((pred_filtered - target_filtered) ** 2)
        
        # Poisson negative log likelihood
        poisson_nll = np.mean(pred_filtered - target_filtered * np.log(pred_filtered + 1e-10))
        
        metrics[name] = {
            'pearson_r': pearson_r,
            'spearman_r': spearman_r,
            'r2': r2,
            'mse': mse,
            'poisson_nll': poisson_nll
        }
    
    return metrics

def plot_and_save_examples(predictions, targets, feature_names, output_dir="eval_plots", n_examples=3):
    os.makedirs(output_dir, exist_ok=True)
    eval_window = slice(325, -325)
    
    for feature_idx, name in enumerate(feature_names):
        plt.figure(figsize=(15, 5))
        
        # Select random examples
        example_indices = np.random.choice(len(predictions), n_examples, replace=False)
        
        for i, idx in enumerate(example_indices):
            plt.subplot(1, n_examples, i+1)
            plt.plot(predictions[idx, feature_idx, eval_window], label='Predicted', alpha=0.8)
            plt.plot(targets[idx, feature_idx, eval_window], label='Actual', alpha=0.8)
            plt.title(f'{name} - Example {i+1}')
            plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{name}_examples.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved example plots for {name} to {plot_path}")

def save_metrics_to_csv(metrics, output_path="evaluation_metrics.csv"):
    # Convert metrics to dataframe
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.index.name = 'feature'
    metrics_df.reset_index(inplace=True)
    
    # Save to CSV
    metrics_df.to_csv(output_path, index=False)
    print(f"Saved evaluation metrics to {output_path}")

def main():
    # Configuration
    model_path = "./newModels/stage2_40000_2.pth"  # Update with your model path
    n_motifs = 9  # Should match your trained model
    test_chromosomes = ["chr8", "chr9"]  # Chromosomes to use for testing
    
    # Feature names (should match training)
    feature_names = [
        "cage_plus", "encodecage_plus", "encoderampage_plus", "grocap_plus", "procap_plus",
        "procap_minus", "grocap_minus", "encoderampage_minus", "encodecage_minus", "cage_minus",
    ]
    
    # Load data and model
    print("Loading model and test data...")
    model, seqs, tars, test_tsses = load_model_and_data(model_path, n_motifs, test_chromosomes)
    
    # Evaluate model
    print(f"Evaluating on {len(seqs)} test sequences...")
    predictions, targets = evaluate_model(model, seqs, tars)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = calculate_metrics(predictions, targets, feature_names)
    
    # Print summary metrics
    print("\nEvaluation Metrics Summary:")
    for feature, vals in metrics.items():
        print(f"\n{feature}:")
        print(f"  Pearson r: {vals['pearson_r']:.4f}")
        print(f"  Spearman r: {vals['spearman_r']:.4f}")
        print(f"  R-squared: {vals['r2']:.4f}")
        print(f"  MSE: {vals['mse']:.4f}")
        print(f"  Poisson NLL: {vals['poisson_nll']:.4f}")
    
    # Save results
    save_metrics_to_csv(metrics)
    plot_and_save_examples(predictions, targets, feature_names)

if __name__ == "__main__":
    main()