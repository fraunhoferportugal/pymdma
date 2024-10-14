from pathlib import Path

import mkdocs_gen_files

SRC_DIR = Path("src")
API_REF_DIR = Path("api_reference")

packages = [
    "pymdma",
]

navigation = mkdocs_gen_files.Nav()


for package in packages:
    package_dir = SRC_DIR / package

    for path in sorted(package_dir.rglob("*.py")):
        module_path = path.relative_to(SRC_DIR).with_suffix("")
        docs_path = module_path.with_suffix(".md")

        parts = tuple(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            docs_path = docs_path.parent / "index.md"

        navigation[parts] = docs_path.as_posix()

        with mkdocs_gen_files.open(API_REF_DIR / docs_path, "w") as docs_file:
            identifier = ".".join(parts)
            docs_file.write(f"::: {identifier}\n")

        mkdocs_gen_files.set_edit_path(API_REF_DIR / docs_path, path)

with mkdocs_gen_files.open(API_REF_DIR / "SUMMARY.md", "w") as navigation_file:
    print("Generating API reference...")
    navigation_file.writelines(navigation.build_literate_nav())

print("API reference generated!")
