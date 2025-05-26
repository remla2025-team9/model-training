import re

README_PATH = "README.md"
COVERAGE_PATH = "pytest-coverage.txt"

# Extract relevant lines from coverage summary
with open(COVERAGE_PATH, "r") as f:
    lines = f.readlines()

coverage_lines = [line.strip() for line in lines if re.match(r"^src/|^TOTAL", line.strip())]

# Build Markdown table
markdown = "| File | Stmts | Miss | Cover |\n"
markdown += "|------|-------|------|--------|\n"

for line in coverage_lines:
    parts = line.split()
    if len(parts) == 4:
        filename, stmts, miss, cover = parts
        label = "**Total**" if filename.startswith("TOTAL") else f"`{filename}`"
        markdown += f"| {label} | {stmts} | {miss} | {cover} |\n"

# Replace section in README.md
with open(README_PATH, "r") as f:
    content = f.read()

start_tag = "<!-- coverage-start -->"
end_tag = "<!-- coverage-end -->"

start_idx = content.find(start_tag)
end_idx = content.find(end_tag)

if start_idx == -1 or end_idx == -1:
    raise ValueError("Missing coverage markers in README.md")

new_content = content[:start_idx + len(start_tag)] + "\n\n" + markdown + "\n" + content[end_idx:]

with open(README_PATH, "w") as f:
    f.write(new_content)

print("✅ README updated with Markdown-formatted coverage table.")
