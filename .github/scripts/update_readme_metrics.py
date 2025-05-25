import json

# Load metrics
with open("reports/evaluation_metrics.json") as f:
    metrics = json.load(f)

# Extract the key values
accuracy = round(metrics["accuracy"], 4)
precision = round(metrics["precision_weighted"], 4)
recall = round(metrics["recall_weighted"], 4)
f1 = round(metrics["f1_weighted"], 4)
cm = metrics["confusion_matrix"]

# Format markdown table
table = f"""| Metric               | Value   |
|----------------------|---------|
| Accuracy             | {accuracy}  |
| Precision (weighted) | {precision}  |
| Recall (weighted)    | {recall}  |
| F1 Score (weighted)  | {f1}  |

<details>
<summary>Confusion Matrix</summary>
{cm}
</details>"""

# Replace section in README.md
with open("README.md", "r") as f:
    content = f.read()

start_tag = "<!-- adequacy-start -->"
end_tag = "<!-- adequacy-end -->"

before = content.split(start_tag)[0] + start_tag + "\n"
after = "\n" + end_tag + content.split(end_tag)[1]
new_content = before + table + after

with open("README.md", "w") as f:
    f.write(new_content)