  import pandas as pd
  import os

  def load_prism_data(csv_dir: str) -> dict:
      """Load MFI, pooling, and README into structured format."""
      data = {}
      for fname in os.listdir(csv_dir):
          if fname.endswith(".csv"):
              data[fname.replace(".csv", "")] = pd.read_csv(os.path.join(csv_dir, fname))
          elif fname == "README.md":
              with open(os.path.join(csv_dir, fname), "r") as f:
                  data["README"] = f.read()
      return data
