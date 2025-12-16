"""
Results Visualization - Create HTML report from workflow results
"""

import json
import csv
from pathlib import Path
from datetime import datetime


def create_html_report(csv_file: str = "workflow_results.csv", 
                       html_file: str = "workflow_results.html"):
    """
    Create an HTML report from the CSV results.
    
    Args:
        csv_file: Path to CSV file with results
        html_file: Path to output HTML file
    """
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        print(f"Error: {csv_file} not found")
        return
    
    # Read CSV
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    if not results:
        print("No results to display")
        return
    
    # Start HTML
    html_content = f"""<!DOCTYPE html>
<html lang="hu">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Workflow Evaluation Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f5f7fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 5px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            color: #333;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .workflow-id {{
            font-weight: bold;
            color: #667eea;
        }}
        .timestamp {{
            font-size: 12px;
            color: #999;
        }}
        .comparison {{
            margin-top: 30px;
        }}
        .comparison h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .chart {{
            margin: 20px 0;
            padding: 20px;
            background: #f5f7fa;
            border-radius: 5px;
        }}
        .workflow-row {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        .workflow-name {{
            width: 200px;
            font-weight: bold;
            color: #333;
        }}
        .bar {{
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 30px;
            border-radius: 3px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 12px;
            min-width: 40px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #999;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 RAG Workflow Evaluation Report</h1>
        
        <div class="summary">
            <div class="metric-card">
                <div class="metric-label">Total Workflows</div>
                <div class="metric-value">{len(results)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Last Update</div>
                <div class="metric-value">{datetime.now().strftime('%H:%M:%S')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Report Date</div>
                <div class="metric-value">{datetime.now().strftime('%Y-%m-%d')}</div>
            </div>
        </div>
        
        <h2>📊 Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Workflow</th>
                    <th>ID</th>
                    <th>Queries</th>
                    <th>Avg Chunks</th>
                    <th>Avg Response Len</th>
                    <th>Avg Utility</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add rows
    for result in results:
        workflow_id = result.get('workflow_id', 'N/A')
        workflow_name = result.get('workflow_name', 'Unknown')
        queries = result.get('queries_processed', 'N/A')
        avg_chunks = result.get('avg_chunks_retrieved', 'N/A')
        avg_response = result.get('avg_response_length', 'N/A')
        avg_utility = result.get('avg_utility_rating', 'N/A')
        timestamp = result.get('timestamp', 'N/A')
        
        # Format timestamp
        if timestamp != 'N/A':
            try:
                ts = datetime.fromisoformat(timestamp)
                timestamp = ts.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
        
        # Format numbers
        if avg_chunks and avg_chunks != 'N/A':
            try:
                avg_chunks = f"{float(avg_chunks):.1f}"
            except:
                pass
        
        if avg_response and avg_response != 'N/A':
            try:
                avg_response = f"{float(avg_response):.0f}"
            except:
                pass
        
        if avg_utility and avg_utility != 'N/A':
            try:
                avg_utility = f"{float(avg_utility):.2f}"
            except:
                pass
        
        html_content += f"""                <tr>
                    <td>{workflow_name}</td>
                    <td class="workflow-id">{workflow_id}</td>
                    <td>{queries}</td>
                    <td>{avg_chunks}</td>
                    <td>{avg_response}</td>
                    <td>{avg_utility}</td>
                    <td class="timestamp">{timestamp}</td>
                </tr>
"""
    
    html_content += """            </tbody>
        </table>
        
        <div class="comparison">
            <h2>📈 Comparison: Average Chunks Retrieved</h2>
            <div class="chart">
"""
    
    # Find max chunks for scaling
    max_chunks = 0
    for result in results:
        try:
            chunks = float(result.get('avg_chunks_retrieved', 0) or 0)
            if chunks > max_chunks:
                max_chunks = chunks
        except:
            pass
    
    if max_chunks == 0:
        max_chunks = 1
    
    # Add bars
    for result in results:
        workflow_name = result.get('workflow_name', 'Unknown')
        workflow_id = result.get('workflow_id', 'N/A')
        try:
            chunks = float(result.get('avg_chunks_retrieved', 0) or 0)
            percentage = (chunks / max_chunks) * 100
            html_content += f"""                <div class="workflow-row">
                    <div class="workflow-name">[{workflow_id}] {workflow_name}</div>
                    <div class="bar" style="width: {percentage}%; min-width: 100px;">
                        {chunks:.1f}
                    </div>
                </div>
"""
        except:
            pass
    
    html_content += f"""            </div>
        </div>
        
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Workflow results are automatically saved and overwritten on each run.</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML
    try:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ HTML report saved to: {html_file}")
        print(f"📊 Open in browser: file:///{Path(html_file).absolute()}")
    except Exception as e:
        print(f"❌ Failed to save HTML: {e}")


def create_markdown_report(csv_file: str = "workflow_results.csv",
                          md_file: str = "workflow_results.md"):
    """
    Create a Markdown report from the CSV results.
    
    Args:
        csv_file: Path to CSV file with results
        md_file: Path to output Markdown file
    """
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        print(f"Error: {csv_file} not found")
        return
    
    # Read CSV
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    if not results:
        print("No results to display")
        return
    
    md_content = f"""# RAG Workflow Evaluation Results

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Workflows**: {len(results)}
- **Last Update**: {datetime.now().isoformat()}

## Detailed Results

| Workflow | ID | Queries | Avg Chunks | Avg Response | Avg Utility | Iterations | Timestamp |
|----------|----|---------|-----------|--------------|-----------|---------| ---|
"""
    
    for result in results:
        workflow_id = result.get('workflow_id', 'N/A')
        workflow_name = result.get('workflow_name', 'Unknown')
        queries = result.get('queries_processed', 'N/A')
        avg_chunks = result.get('avg_chunks_retrieved', 'N/A')
        avg_response = result.get('avg_response_length', 'N/A')
        avg_utility = result.get('avg_utility_rating', 'N/A')
        avg_iterations = result.get('avg_iterations', 'N/A')
        timestamp = result.get('timestamp', 'N/A')
        
        # Format numbers
        if avg_chunks and avg_chunks != 'N/A':
            try:
                avg_chunks = f"{float(avg_chunks):.1f}"
            except:
                avg_chunks = '-'
        else:
            avg_chunks = '-'
        
        if avg_response and avg_response != 'N/A':
            try:
                avg_response = f"{float(avg_response):.0f}"
            except:
                avg_response = '-'
        else:
            avg_response = '-'
        
        if avg_utility and avg_utility != 'N/A':
            try:
                avg_utility = f"{float(avg_utility):.2f}"
            except:
                avg_utility = '-'
        else:
            avg_utility = '-'
        
        if avg_iterations and avg_iterations != 'N/A':
            try:
                avg_iterations = f"{float(avg_iterations):.1f}"
            except:
                avg_iterations = '-'
        else:
            avg_iterations = '-'
        
        # Format timestamp
        if timestamp != 'N/A':
            try:
                ts = datetime.fromisoformat(timestamp)
                timestamp = ts.strftime('%H:%M:%S')
            except:
                pass
        
        md_content += f"| {workflow_name} | {workflow_id} | {queries} | {avg_chunks} | {avg_response} | {avg_utility} | {avg_iterations} | {timestamp} |\n"
    
    md_content += f"""

## Workflow Descriptions

### [11] Feedback Loop RAG
Tracks relevance feedback on retrieved chunks and adjusts future retrieval scores based on feedback history.

### [12] Adaptive RAG
Classifies queries by type (Factual, Analytical, Opinion, Contextual) and adjusts retrieval strategy accordingly.

### [13] Self-RAG
Implements self-reflective RAG with iterative refinement and relevance filtering.

### [14] Proposition Chunking RAG
Decomposes chunks into atomic propositions for fine-grained retrieval at proposition level.

### [19] HyDE RAG
Generates hypothetical documents to improve semantic matching using hypothetical embeddings.

## Notes

- Results are automatically saved to `workflow_results.csv`
- Each run overwrites previous results with fresh metrics
- Metrics are calculated from all processed queries
- Timestamps show when each workflow was executed

---
*Report generated: {datetime.now().isoformat()}*
"""
    
    try:
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"✅ Markdown report saved to: {md_file}")
    except Exception as e:
        print(f"❌ Failed to save Markdown: {e}")


if __name__ == "__main__":
    print("Creating visualization reports...")
    create_html_report()
    create_markdown_report()
    print("✅ All reports generated!")
