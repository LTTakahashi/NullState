import pandas as pd
import anndata as ad
from pathlib import Path
import json
import logging
import yaml
import re

def count_offtarget_mesenchyme(query: ad.AnnData, class_column: str = 'pilot_class', label_column: str = 'pred_label', pattern: str = 'mesench|stroma|fibro', protocol_column: str = 'protocol') -> dict:
    """Counts true_offtarget cells matching mesenchymal pattern."""
    # Ensure protocol column exists
    if protocol_column not in query.obs.columns:
        # Fallback to dataset id or similar if protocol is missing
        fallback_cols = ['id', 'dataset_id', 'batch']
        for col in fallback_cols:
            if col in query.obs.columns:
                protocol_column = col
                logging.warning(f"'protocol' column missing. Using '{col}' instead.")
                break
        else:
            query.obs['dummy_protocol'] = 'unknown'
            protocol_column = 'dummy_protocol'
            
    mask = (query.obs[class_column] == 'true_offtarget') & \
           (query.obs[label_column].str.contains(pattern, flags=re.IGNORECASE, na=False))
           
    mes = query[mask]
    
    by_protocol = mes.obs[protocol_column].value_counts().to_dict()
    matched_labels = mes.obs[label_column].unique().tolist()
    
    return {
        "total_count": int(mes.n_obs),
        "by_protocol": by_protocol,
        "matched_labels": matched_labels
    }

def evaluate_gate(counts: dict, green_total: int = 1000, green_per_protocol: int = 300, yellow_total: int = 300) -> dict:
    """Applies hard thresholds for the count check gate."""
    total = counts["total_count"]
    by_protocol = counts["by_protocol"]
    
    protocols_above_threshold = sum(1 for v in by_protocol.values() if v >= green_per_protocol)
    
    if total >= green_total and protocols_above_threshold >= 2:
        color = "GREEN"
        reason = f"Total N={total} (>= {green_total}) and {protocols_above_threshold} protocols have N >= {green_per_protocol}."
        msg = "Proceed to Aim 3. Full power available."
    elif total >= yellow_total:
        color = "YELLOW"
        reason = f"Total N={total} (>= {yellow_total}) but only {protocols_above_threshold} protocols meet threshold."
        msg = "Proceed to Aim 3, but scope-limited (pooled comparisons only)."
    else:
        color = "RED"
        reason = f"Total N={total} (< {yellow_total})."
        msg = "Stop. Diagnostic required."
        
    return {
        "gate": "Count Check",
        "decision": color,
        "reason": reason,
        "action": msg
    }

def run_red_diagnostic(query: ad.AnnData, label_column: str = 'pred_label', class_column: str = 'pilot_class', pattern: str = 'mesench|stroma|fibro') -> dict:
    """Diagnoses where candidate mesenchyme went if counts are low."""
    logging.info("Running RED diagnostic...")
    
    mask = query.obs[label_column].str.contains(pattern, flags=re.IGNORECASE, na=False)
    cand = query[mask]
    
    class_dist = cand.obs[class_column].value_counts().to_dict()
    
    # Determine likely cause
    ambiguous = class_dist.get('ambiguous_novel', 0)
    total_cand = sum(class_dist.values())
    
    if total_cand == 0:
        diagnosis = "Candidate mesenchyme is genuinely scarce in the raw data. Biology limit."
    elif ambiguous / total_cand > 0.5:
        diagnosis = "Most candidate mesenchyme landed in 'ambiguous_novel'. Reference likely lacks adequate mesodermal representation. Fixable by broadening fetal reference."
    else:
        diagnosis = "Cells spread across other classes. Check threshold calibrations and origin map."
        
    return {
        "total_candidates": int(total_cand),
        "class_distribution": class_dist,
        "diagnosis": diagnosis
    }

def run_count_check(query: ad.AnnData, config_path: str = 'config/params.yaml') -> dict:
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
        
    pattern = params.get('mesenchymal_pattern', "mesench|stroma|fibro")
    green_total = params.get('count_green_total', 1000)
    green_per_protocol = params.get('count_green_per_protocol', 300)
    yellow_total = params.get('count_yellow_total', 300)
    
    counts = count_offtarget_mesenchyme(query, pattern=pattern)
    gate = evaluate_gate(counts, green_total, green_per_protocol, yellow_total)
    
    diagnostic = None
    if gate["decision"] == "RED":
        diagnostic = run_red_diagnostic(query, pattern=pattern)
        
    return {
        "counts": counts,
        "gate": gate,
        "diagnostic": diagnostic
    }

def save_results(results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "count_gate.json", 'w') as f:
        json.dump(results["gate"], f, indent=2)
        
    with open(output_dir / "counts.json", 'w') as f:
        json.dump(results["counts"], f, indent=2)
        
    if results.get("diagnostic"):
        with open(output_dir / "red_diagnostic.json", 'w') as f:
            json.dump(results["diagnostic"], f, indent=2)
