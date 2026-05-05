import os
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader

def extract_auc_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text = page.extract_text()
        if "AUC =" in text:
            for line in text.splitlines():
                if "AUC =" in line:
                    try:
                        return float(line.split("AUC =")[-1].strip())
                    except:
                        return None
    return None

def extract_sens_spec_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        if row["sens"] >= 0.95:
            return row["sens"], row["spec"]
    return None, None

base_dir = os.getcwd()
output_file = os.path.join(base_dir, "summary_results.csv")

with open(output_file, "w") as out:
    out.write("Features,Execution,AUC,Sens,Spec\n")

    for num in range(1, 7):
        print(f"📁 Processing {num} features")
        dir_path = os.path.join(base_dir, str(num))
        if not os.path.isdir(dir_path):
            continue

        for feat_combo in sorted(os.listdir(dir_path)):
            feat_path = os.path.join(dir_path, feat_combo)
            if not os.path.isdir(feat_path):
                continue

            aucs, sens_list, spec_list = [], [], []

            for i in range(10):
                run_dir = os.path.join(feat_path, str(i))
                if not os.path.isdir(run_dir):
                    continue

                pdf_path = os.path.join(run_dir, f"hcpa-{i}.pdf")
                csv_path = os.path.join(run_dir, f"hcpa-{i}-thresholds.csv")

                auc = extract_auc_from_pdf(pdf_path)
                sens, spec = extract_sens_spec_from_csv(csv_path)

                if auc is not None and sens is not None and spec is not None:
                    aucs.append(auc)
                    sens_list.append(sens)
                    spec_list.append(spec)
                    out.write(f"{feat_combo},{i},{auc:.4f},{sens:.4f},{spec:.4f}\n")

print(f"\n✅ Summary written to: {output_file}")
