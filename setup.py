from pathlib import Path

from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).parent

with open("README.md") as fh:
    long_description = fh.read()


def find_requirements(filename):
    with (ROOT / "requirements" / filename).open() as f:
        return [s for s in [line.strip(" \n") for line in f] if not s.startswith("#") and s != ""]


def find_version():
    if Path("VERSION").exists():
        with open("VERSION") as f:
            return f.read().strip()
    return "0.0.0"


def parse_extra_requires(base_requires, extras=["image", "time_series", "tabular"]):
    extra_requires = {}
    for extra in extras:
        extra_requires[extra] = find_requirements(f"requirements-{extra}.txt")
    # remove any repeated dependencies
    # this happens because poetry always adds the base dependencies to optional dependencies on export
    repeated = set(base_requires)
    for key, value in extra_requires.items():
        extra_requires[key] = list(set(value) - repeated)
        repeated.update(extra_requires[key])

    # add all dependencies to the "all" extra
    # extra_requires["all"] = []
    # for extra in extra_requires.values():
    #     extra_requires["all"].extend(extra)
    return extra_requires


base_requires = find_requirements("requirements.txt")
extra_requires = parse_extra_requires(base_requires)

packs = find_namespace_packages(where="src", exclude=["**/tests/*", "*.text*", "*.api*"])
package_dirs = {
    "": "src",
}


setup(
    name="pymdma",
    version=find_version(),
    author="Fraunhofer Portugal",
    # author_email="marilia.barandas@aicos.fraunhofer.pt",
    description="Multimodal Data Metrics for Auditing real and synthetic data",
    long_description=long_description,
    url="https://github.com/fraunhoferportugal/pymdma",
    long_description_content_type="text/markdown",
    python_requires=">=3.9.0",
    install_requires=base_requires,
    extras_require=extra_requires,
    packages=packs,
    package_dir=package_dirs,
    license="LGPL-3.0-or-later",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
)
