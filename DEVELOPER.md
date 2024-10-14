# Developer Notes

## Packaging and Dependencies

This project uses [Conda](https://anaconda.org/anaconda/python) to manage Python virtual environments and [Poetry](https://python-poetry.org/) as the main packaging and dependency manager.

Poetry allows to organize the dependencies into groups so you can isolate and optimize the dependency trees for different purposes. A dependency can be added with:

```
poetry add <package>
```

To add a dependency into a specific group, use:

```
poetry add <package> --group <group>
```

A dependency group can be made optional (i.e. not installed by default unless explicitly included) can be done by adding the following into `pyproject.toml` file above the group dependencies:

```
[tool.poetry.group.<group>]
optional = true
[tool.poetry.group.<group>.dependencies]
...
```

Poetry allows for more advanced specification of dependencies, such as version ranges, git repositories, etc. For more information, please refer to the [Poetry documentation](https://python-poetry.org/docs/dependency-specification/).

A group of dependencies can be installed with:

```
# Installs non-optional dependencies and dependencies in the with parameter
poetry install --with <group>,<group>,...
```

It's also possible to install a single group isolated with:

```
poetry install --only <group>,<group>,...
```

A coding standard is enforced using [Black](https://github.com/psf/black), [isort](https://pypi.org/project/isort/) and
[Flake8](https://flake8.pycqa.org/en/latest/). Python 3 type hinting is validated using
[MyPy](https://pypi.org/project/mypy/).

Unit tests are written using [Pytest](https://docs.pytest.org/en/latest/), documentation is written
using [Google Style Python Docstring](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
[Pydocstyle](http://pydocstyle.org/) is used as static analysis tool for checking compliance with Python docstring
conventions.

Additional code security standards are enforced by [Safety](https://github.com/pyupio/safety) and
[Bandit](https://bandit.readthedocs.io/en/latest/). [Git-secrets](https://github.com/awslabs/git-secrets)
ensure you're not pushing any passwords or sensitive information into your Bitbucket repository.
Commits are rejected if the tool matches any of the configured regular expression patterns that indicate that sensitive
information has been stored improperly.

We use [sphinx](https://www.sphinx-doc.org) or [mkdocs](https://www.mkdocs.org) for building documentation (we favour usage of `mkdocs` as `sphinx` is currently not fully supported out-of-the-box).
You can call `make build_docs` from the project root, the docs will be built under `docs/_build/html`.
Detail information about documentation can be found [here](docs/index.md).

## Git Hooks

We rely on [pre-commit](https://pre-commit.com) hooks to ensure that the code is properly-formatted, clean, and type-safe when it's checked in.
The `run install` step described below installs the project pre-commit hooks into your repository. These hooks
are configured in [`.pre-commit-config.yaml`](/.pre-commit-config.yaml). After installing the development requirements
and cloning the package, run

```
pre-commit install
```

from the project root to install the hooks locally.  Now before every `git commit ...` these hooks will be run to verify
that the linting and type checking is correct. If there are errors, the commit will fail, and you will see the changes
that need to be made. Alternatively, you can run pre-commit

```
pre-commit run --all-files  
```

If necessary, you can temporarily disable a hook using Git's `--no-verify` switch. However, keep in mind that the CI
build enforces these checks, so the build will fail.

You can build your own pre-commit scripts. Put them on `scripts` folder. To make a shell script executable, use the
following command.

```
git update-index --chmod=+x scripts/name_of_script.sh
```

Don’t forget to commit and push your changes after running it!

**Warning:** You need to run `git commit` with your conda environment activated. This is because by default the packages used
by pre-commit are installed into your project's conda environment. (note: `pre-commit install --install-hooks` will install
the pre-commit hooks in the currently active environment).

### Markdown

Local links can be written as normal, but external links should be referenced at the  bottom of the Markdown file for clarity.
For example:

```md
Use a local link to reference the [`README.md`](../README.md) file, but an external link for [Fraunhofer AICOS][fhp-aicos].

[fhp-aicos]: https://www.fraunhofer.pt/
```

We also try to wrap Markdown to a line length of 88 characters. This is not strictly
enforced in all cases, for example with long hyperlinks.

## Testing

\[Tests are written using the `pytest` framework\]\[pytest\], with its configuration in the `pyproject.toml` file.
Note, only tests in `time_series_metrics/tests` folders folder are run.
To run the tests, enter the following command in your terminal:

```shell
pytest -vvv
```

### Code coverage

\[Code coverage of Python scripts is measured using the `coverage` Python package\]\[coverage\]; its configuration
can be found in `pyproject.toml`.
To run code coverage, and view it as an HTML report, enter the following command in your terminal:

```shell
coverage run -m pytest
coverage html
```

or use the `make` command:

```shell
make coverage_html
```

The HTML report can be accessed at `htmlcov/index.html`.

## Set private environment variables in .envrc file

System specific variables (e.g. absolute paths to datasets) should not be under version control, or it will result in
conflict between different users. Your private keys also shouldn't be versioned since you don't want them to be leaked.

The .env file, which serves as an example. Create a new file called .env (this name is excluded from version control in
.gitignore). You should use it for storing environment variables like this:

```
MY_VAR=/home/user/my_system_path
```

All variables from .env are loaded in config.py automatically.

## Version control your data and models with DVC

Use DVC to version control big files, like your data or trained ML models. To initialize the dvc repository:

```
dvc init
```

To start tracking a file or directory, use dvc add (e.g. pictures):

```
dvc add data/raw/*.jpg
```

DVC stores information about the added file (or a directory) in a special .dvc file named data/raw/\*jpg.dvc, a small text
file with a human-readable format. This file can be easily versioned like source code with Git, as a placeholder for the
original data:

```
git add data/raw/*jpg.dvc
git commit -m "Add raw data"
```

We recommend tagging each time you modify the files inside the data folder

```
git commit -m "Add more images. Model trained with 2000 images."
git tag -a "v2.0" -m "model v2.0, 2000 images"
git push --tags
dvc push  # Upload dataset to S3 Bucket on Minio Server
```

The regular workflow is to use `git checkout` first to switch a branch, checkout a commit/tag, or a revision of a .dvc file,
and then run `dvc checkout` to sync data: To switch to a previous version (e.g. with tag v1.0) of our code and data.
DVC checkout will remove the new files.

```
git checkout v1.0
dvc checkout
```

Read more in the [docs](https://dvc.org/doc/start/data-versioning)!

## Hydra

Hydra is an open-source Python framework that simplifies the development of research and other complex applications.
The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through
config files and the command line. The name Hydra comes from its ability to run multiple similar jobs - much like a
Hydra with multiple heads.

We recommend going through at least the [Basic Tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli),
and the docs about [Instantiating objects with Hydra](https://hydra.cc/docs/patterns/instantiate_objects/overview).

## CI

All PRs trigger a CI job to run linting, type checking, tests, and build docs. The CI script is located [here](Jenkinsfile)
and should be considered the source of truth for running the various development commands.

## Line Endings

The [`.gitattributes`](.gitattributes) file controls line endings for the files in this repository.

## Prerequisites

Nearly all prerequisites are managed by Conda. All you need to do is make sure that you have a working Python 3
environment and install miniconda itself. Conda manages `virtualenvs` as well. Typically, on a project that uses virtualenv
directly you would activate the virtualenv to get all the binaries that you install with pip onto the path.
Conda works in a similar way but with different commands.

Use miniconda for your python environments (it's usually unnecessary to install full anaconda environment, miniconda
should be enough). It makes it easier to install some dependencies, like `cudatoolkit` for GPU support. It also allows you
to access your environments globally.

Example installation:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

## IDE Setup

There are a few useful plugins that are probably available for most IDEs. Using Pycharm, you'll want to install the
black plugin.

- [blackconnect](https://plugins.jetbrains.com/plugin/14321-blackconnect) can be configured to auto format files on save.
  Just run `make blackd` from a shell to set up the server and the plugin will do its thing. You need to configure it to
  format on save, it's off by default.

## Development Details

You can run `make help` for a full list of targets that you can run. These are the ones that you'll need most often.

```bash
# For running tests locally
make test

# For formatting and linting
make lint
make format
make format-fix

# Remove all generated artifacts
make clean
```

## Reproducible environment

The first step in reproducing an analysis is always reproducing the computational environment it was run in.
**You need the same tools, the same libraries, and the same versions to make everything play nicely together.**

By listing all of your requirements in the repository you can easily track the packages needed to recreate the analysis,
but what tool should we use to do that?

Whilst popular for scientific computing and data-science, [conda](https://docs.conda.io/en/latest/) poses problems for collaboration and packaging:

- It is hard to reproduce a conda-environment across operating systems
- It is hard to make your environment "pip-installable" if your environment is fully specified by conda

### Files

Due to these difficulties, we recommend only using conda to create a virtual environment and list dependencies not available through usage of Poetry.

- `environment.yaml` - Defines the base conda environment and any dependencies not managed by Poetry.
- `pyproject.toml` - This is the preferred way to specify dependencies. If you need to add a dependency, chances are it goes here!
- `requirements/` - Folder containing requirements files in case the environment where the code is deployed can't run Poetry. These can be generated via Poetry or using the `Makefile` rules provided.
