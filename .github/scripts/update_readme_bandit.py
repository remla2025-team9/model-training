import json
from pathlib import Path

BANDIT_OUTPUT_PATH = "bandit_output.json"
README_PATH = "README.md"
TAG_START = "<!-- BANDIT_START -->"
TAG_END = "<!-- BANDIT_END -->"


def load_bandit_results(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_summary(bandit_data):
    metrics = bandit_data.get("metrics", {})
    results = bandit_data.get("results", [])

    # Initialize file summaries with loc and zero counts
    file_summaries = {
        file: {
            "file": file,
            "loc": data.get("loc", 0),
            "high": 0,
            "medium": 0,
            "low": 0
        }
        for file, data in metrics.items()
        if file != "_totals"
    }

    # Count severities per file
    for result in results:
        file = result["filename"]
        severity = result["issue_severity"].lower()
        if file in file_summaries:
            if severity in file_summaries[file]:
                file_summaries[file][severity] += 1

    return list(file_summaries.values())


def format_bandit_section(summary):
    lines = [
        TAG_START,
        "## Bandit Security Analysis",
        "",
        "| File | LOC | High | Medium | Low |",
        "|------|-----|------|--------|-----|",
    ]
    for entry in sorted(summary, key=lambda e: e["file"]):
        lines.append(
            f"| `{entry['file']}` | {entry['loc']} | {entry['high']} | {entry['medium']} | {entry['low']} |"
        )

    lines.append(TAG_END)
    return "\n".join(lines)


def update_readme(readme_path, new_section):
    readme = Path(readme_path)
    if not readme.exists():
        readme.write_text(new_section)
        return

    content = readme.read_text()
    if TAG_START in content and TAG_END in content:
        pre = content.split(TAG_START)[0]
        post = content.split(TAG_END)[1]
        updated = f"{pre}{new_section}{post}"
    else:
        updated = content.strip() + "\n\n" + new_section

    readme.write_text(updated)


def main():
    bandit_data = load_bandit_results(BANDIT_OUTPUT_PATH)
    summary = extract_summary(bandit_data)
    section = format_bandit_section(summary)
    update_readme(README_PATH, section)


if __name__ == "__main__":
    main()
