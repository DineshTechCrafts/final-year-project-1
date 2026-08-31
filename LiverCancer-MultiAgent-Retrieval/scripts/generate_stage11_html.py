import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    results_dir = PROJECT_ROOT / "results" / "stage11"
    
    # Load required data
    with open(results_dir / "stage11_summary.json", "r") as f:
        summary = json.load(f)
        
    with open(results_dir / "stage11_ablation.json", "r") as f:
        ablation = json.load(f)
        
    with open(results_dir / "stage11_benchmark.json", "r") as f:
        bench = json.load(f)
        
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Stage 11 Final Research Validation</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; line-height: 1.6; color: #333; }}
        h1 {{ border-bottom: 2px solid #005A9C; padding-bottom: 10px; color: #005A9C; }}
        h2 {{ color: #005A9C; margin-top: 30px; }}
        .alert {{ background: #ffeeba; color: #856404; padding: 15px; border-left: 5px solid #ffc107; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
    </script>
</head>
<body>

    <h1>Liver Cancer Image Retrieval: Final Validation Report</h1>
    
    <div class="alert">
        <h2>CRITICAL SCIENTIFIC LIMITATIONS</h2>
        <ul>
            {"".join([f"<li>{l}</li>" for l in summary["scientific_limitations"]])}
        </ul>
    </div>

    <h2>1. System Architecture</h2>
    <div class="mermaid">
    graph TD
        A[Medical Image] --> B[Preprocessing]
        B --> C[U-Net Segmentation]
        C --> D[Clinical/Anatomical Features]
        C --> E[Visual Embeddings]
        D --> F[Multimodal Fusion]
        E --> F
        F --> G[FAISS Retrieval]
        G --> H[Multi-Agent Evidence Reranking]
        H --> I[Explainable Retrieval Report]
    </div>

    <h2>2. Ablation Study (Computational Relevance Proxy)</h2>
    <p><i>Note: 95% Confidence Intervals calculated via B=1000 Patient-Level Bootstrap Resampling on TEST partition.</i></p>
    <table>
        <tr>
            <th>Method</th>
            <th>Recall@10 (Mean)</th>
            <th>95% CI Lower</th>
            <th>95% CI Upper</th>
        </tr>
"""
    
    for method, res in ablation["results"].items():
        html += f"""
        <tr>
            <td>{method}</td>
            <td>{res['mean']:.3f}</td>
            <td>{res['ci_95'][0]:.3f}</td>
            <td>{res['ci_95'][1]:.3f}</td>
        </tr>
"""

    html += f"""
    </table>
    
    <h2>3. Computational Latency</h2>
    <table>
        <tr>
            <th>Pipeline Stage</th>
            <th>Latency</th>
        </tr>
"""
    for stage, time_str in bench.items():
        html += f"""
        <tr>
            <td>{stage}</td>
            <td>{time_str}</td>
        </tr>
"""
        
    html += """
    </table>
</body>
</html>
"""
    
    with open(results_dir / "stage11_final_report.html", "w") as f:
        f.write(html)
        
    print("Stage 11 HTML Report generated successfully.")

if __name__ == "__main__":
    main()
