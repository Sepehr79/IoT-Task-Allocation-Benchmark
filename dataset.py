from pathlib import Path

dataset_path = Path("dataset")

sizes = []

for file in dataset_path.rglob("*"):
    if file.is_file():
        sizes.append(file.stat().st_size)

if sizes:
    print(f"Number of files: {len(sizes)}")
    print(f"Mean: {sum(sizes) / len(sizes) / 1024:.2f} KB")
    print(f"Min : {min(sizes) / 1024:.2f} KB")
    print(f"Max : {max(sizes) / 1024:.2f} KB")
else:
    print("No files found.")