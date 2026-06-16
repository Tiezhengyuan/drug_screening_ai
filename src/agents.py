import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class DataIngestionAgent:
    """Processes CSV input files (MFI, treatment info, pooling, readme)."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def load_files(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from data directory."""
        data = {}
        for csv_file in self.data_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            data[csv_file.stem] = df
            print(f"Loaded {csv_file.name}: {len(df)} rows, {len(df.columns)} columns")
        return data
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Basic validation of loaded data."""
        # Check required columns exist
        for name, df in data.items():
            if "MFI" in df.columns or "mfi" in df.columns:
                mfi_col = "MFI" if "MFI" in df.columns else "mfi"
                if df[mfi_col].isna().sum() > 0:
                    print(f"Warning: {name} has missing MFI values")
            if "treatment" in df.columns or "Treatment" in df.columns:
                treat_col = "treatment" if "treatment" in df.columns else "Treatment"
                if df[treat_col].isna().sum() > 0:
                    print(f"Warning: {name} has missing treatment values")
        return True


class DrugSensitivityAgent:
    """Analyzes drug sensitivity from MFI data."""
    
    def analyze_sensitivity(self, data: pd.DataFrame, 
                           mfi_col: str = "MFI", 
                           treatment_col: str = "treatment",
                           control_col: str = "control") -> Dict:
        """
        Calculate drug sensitivity metrics.
        
        Args:
            data: DataFrame with MFI values
            mfi_col: Column name for MFI (Mean Fluorescence Intensity)
            treatment_col: Column name for treatment/drug identifier
            control_col: Column name for control group identifier
        """
        results = {
            "drugs": [],
            "log_fold_change": [],
            "viability_percent": [],
            "sensitive": []
        }
        
        # Group by treatment
        for treatment in data[treatment_col].unique():
            treatment_data = data[data[treatment_col] == treatment]
            control_data = data[data[control_col] == True] if control_col in data.columns else None
            
            if len(treatment_data) > 0:
                mfi_mean = treatment_data[mfi_col].mean()
                mfi_std = treatment_data[mfi_col].std()
                
                # Calculate viability relative to control
                if control_data is not None and len(control_data) > 0:
                    control_mean = control_data[mfi_col].mean()
                    viability = (mfi_mean / control_mean) * 100 if control_mean > 0 else 0
                    log_fc = np.log2(mfi_mean / control_mean) if control_mean > 0 else 0
                else:
                    viability = 0
                    log_fc = 0
                
                results["drugs"].append(treatment)
                results["viability_percent"].append(viability)
                results["log_fold_change"].append(log_fc)
                results["sensitive"].append(viability < 30)  # <30% viability = sensitive
        
        result_df = pd.DataFrame(results)
        return {
            "summary": result_df,
            "sensitive_drugs": result_df[result_df["sensitive"]]["drugs"].tolist(),
            "most_sensitive": result_df.loc[result_df["viability_percent"].idxmin()] if len(result_df) > 0 else None
        }


class PRISMAgent:
    """Main agent that orchestrates the workflow."""
    
    def __init__(self, rag_retriever, data_dir: Path):
        self.rag = rag_retriever
        self.ingestion_agent = DataIngestionAgent(data_dir)
        self.sensitivity_agent = DrugSensitivityAgent()
        
    def process_query(self, query: str, data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
        """Process a user query through the multi-agent workflow."""
        # Step 1: Load data if not provided
        if data is None:
            data = self.ingestion_agent.load_files()
        
        results = {"query": query, "data_loaded": list(data.keys())}
        
        # Step 2: Run sensitivity analysis on each dataset
        sensitivity_results = {}
        for name, df in data.items():
            if "MFI" in df.columns or "mfi" in df.columns:
                mfi_col = "MFI" if "MFI" in df.columns else "mfi"
                # Determine treatment column
                treat_col = next((c for c in df.columns if "treatment" in c.lower()), None)
                if treat_col:
                    sensitivity_results[name] = self.sensitivity_agent.analyze_sensitivity(
                        df, mfi_col=mfi_col, treatment_col=treat_col
                    )
        
        results["sensitivity_analysis"] = sensitivity_results
        
        # Step 3: Retrieve relevant literature for sensitive drugs
        literature_context = []
        for name, analysis in sensitivity_results.items():
            for drug in analysis.get("sensitive_drugs", [])[:5]:  # Top 5 sensitive drugs
                context = self.rag.retrieve(f"{drug} drug mechanism of action PRISM", top_k=2)
                literature_context.append({
                    "drug": drug,
                    "context": context
                })
        
        results["literature_context"] = literature_context
        
        # Step 4: Generate summary
        summary_lines = ["=== PRIM Drug Screening Results ===\n"]
        for name, analysis in sensitivity_results.items():
            sensitive = analysis.get("sensitive_drugs", [])
            summary_lines.append(f"\nDataset: {name}")
            summary_lines.append(f"  Sensitive drugs (<30% viability): {len(sensitive)}")
            if sensitive:
                summary_lines.append(f"  Top candidates: {', '.join(sensitive[:5])}")
                if analysis.get("most_sensitive") is not None:
                    ms = analysis["most_sensitive"]
                    summary_lines.append(f"  Most sensitive: {ms['drugs']} (viability: {ms['viability_percent']:.1f}%)")
        
        results["summary"] = "\n".join(summary_lines)
        return results