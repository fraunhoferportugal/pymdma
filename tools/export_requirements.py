import argparse
from typing import Dict, List, Union

import toml


def parse_args():
    parser = argparse.ArgumentParser(description="Export requirements from pyproject.toml")
    parser.add_argument("pyproject_toml", type=str, help="Path to pyproject.toml")
    parser.add_argument("output", type=str, help="Path to output requirements.txt")
    parser.add_argument("--group_names", type=str, nargs="+", default=[], help="Group names to export")
    return parser.parse_args()


def raise_major(version: str):
    splitted = version[1:].split(".")
    version_values = [int(s) for s in splitted if s.isnumeric()]
    if len(version_values) < 3:
        version_values.extend([0] * (3 - len(version_values)))

    parsed = version_values.copy()
    for i, v in enumerate(version_values):
        if v != 0:
            if i + 1 < len(version_values) and version_values[i + 1] != 0:
                parsed[i] = v + 1
                parsed[i + 1 :] = [0] * (len(version_values) - i - 1)
                break
            else:
                parsed[i] = v + 1
                break
        if v == 0 and i == len(version_values) - 1:
            parsed[i] = 1
            break

    assert parsed != version_values, f"Invalid version format {version}"
    # ensure major.minor.patch
    return f" >= {'.'.join(map(str, version_values))}, < {'.'.join(map(str, parsed))}"


def parse_dependency(dependency: Union[Dict[str, str], str]) -> List[str]:
    if isinstance(dependency, str):
        if dependency.startswith("^"):
            return [raise_major(dependency)]
        return [dependency]
    if isinstance(dependency, dict):
        if "version" in dependency:
            version = dependency["version"]
            if version.startswith("^"):
                version = raise_major(version)
            extras = dependency.get("extras", None)
            if extras is None:
                return [version]
            else:
                return [f"[{extra}] {version}" for extra in extras]
        if "git" in dependency:
            return [f" @ git+{dependency['git']}"]
        raise ValueError(f"Dependency is not a valid format: {dependency}")


if __name__ == "__main__":

    args = parse_args()
    with open(args.pyproject_toml) as f:
        data = toml.load(f)

    poetry = data["tool"]["poetry"]
    dependencies = {
        "base": poetry["dependencies"],
    }

    for group_name in args.group_names:
        dependencies[group_name] = poetry["group"][group_name]["dependencies"]

    for group_name, group_dependencies in dependencies.items():
        with open(f"{args.output}_{group_name}.txt", "w") as f:
            for dep_name, dep_value in group_dependencies.items():
                print(dep_value, parse_dependency(dep_value))
                for version_spec in parse_dependency(dep_value):
                    # print(version_spec)
                    f.write(f"{dep_name}{version_spec}\n")

    print(data["tool"]["poetry"]["dependencies"].keys())
    print(data["tool"]["poetry"]["group"]["image"]["dependencies"])
    print(data["tool"]["poetry"]["group"]["prod"]["dependencies"])
