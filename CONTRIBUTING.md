# Contributing

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or
additional documentation, we greatly value feedback and contributions.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution. Any contributions should come through valid Pull/Merge.

## Reporting Bugs/Feature Requests

When filing an issue, please check previous issues to make sure somebody else hasn't already reported the issue.
Please try to include as much information as you can. Details like these are incredibly useful:

- A reproducible test case or series of steps
- The version of our code being used
- Any modifications you've made relevant to the bug
- Anything unusual about your environment or deployment

## Contributing via Pull Requests

Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the **dev** branch.
1. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.

To send a pull request, please:

1. Fork the repository.
1. Modify the source; please focus on the specific change you are contributing.
   If you also reformat all the code, it will be hard for us to focus on your change.
1. Ensure local tests pass by executing `pytest`.
1. Commit to your fork using clear commit messages.
1. Send us a pull request, answering any default questions in the pull request interface.

## Tips for Modifying the Source Code

- We recommend developing on Linux as this is the only OS where all features are currently 100% functional.
- Use **Python >= 3.11.0** for development.
- Please try to avoid introducing additional dependencies on 3rd party packages.
- We encourage you to add your own unit tests, but please ensure they run quickly (unit tests should train models on
  small data-subsample with the lowest values of training iterations and time-limits that suffice to evaluate the intended
  functionality).

## Contribution Guidelines

1. Please adhere to the [PEP-8](https://www.python.org/dev/peps/pep-0008/) standards. A maximum line length of 120 characters is
   allowed for consistency with the C++ code. Editors such as PyCharm [(see here)](https://www.jetbrains.com/help/pycharm/code-inspection.html)
   and Visual Studio Code [(see here)](https://code.visualstudio.com/docs/python/linting#_flake8) can be configured to check
   for PEP8 issues.
1. Code can be validated with flake8 using the configuration file in the root directory called `.flake8`.
1. Run `black` and `isort` linters before creating a PR.
1. Any changes to core functionality must pass all existing unit tests.
1. Functions/methods should have type hints for arguments and return types.
1. Additional functionality should have associated unit tests.
1. Provide documentation ([Numpydoc Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html) and [Griffe Numpydoc-style](https://mkdocstrings.github.io/griffe/reference/docstrings/#numpydoc-style)) for any new functionality.
   whenever possible, even for simple functions or classes.

### Design patterns

Favour the following design patterns, in this order:

1. Functional programming: using functions to do one thing well, not altering the original data.
1. Modular code: using functions to eliminate duplication, improve readability and reduce indentation.
1. Object-oriented programming: Generally avoid, unless you are customising an API (for example `DataFrame`) or defining your own API.

If you are not, at least, adhering to a modular style then you have gone very wrong.
You should implement unit tests for each of your functions, something which is generally more tricky for object-oriented programming.

### Programming

1. Don't compare boolean values to `True` or `False`.
1. Favour `is not condition` over `not condition is`
1. Don't compare a value to `None` (`value == None`), always favour `value is None`
1. Favour [`logging`](https://docs.python.org/3/howto/logging.html) over `print`
1. Favour using configuration files, or (faster/lazier/less good/ok) `GLOBAL_VARIABLES` near the top of your code, rather than repeated
   use of hard-coded variables in your code, particularly when with file path variables, but also for repeated numeric hyperparameters.

### Naming conventions

1. Functions / methods: `function`, `my_function` (snake case)
1. Variables / attributes: `variable`, `my_var` (snake case)
1. Class: `Model`, `MyClass` (camel case)
1. Module / file names / directory names: `module`, `file_name.py`, `dir_name` (snake case)
1. Global\* / constants: `A_GLOBAL_VARIABLE` (screaming snake case)
1. Keep all names as short and descriptive as possible. Variable names such as `x` or `df` are highly discouraged unless they are genuinely
   representing abstract concepts.
1. Favour good naming conventions over helpful comments

## Guidelines for creating a good pull request

1. A PR should describe the change clearly and most importantly it should mention the motivation behind the change.
1. If the PR is fixing a performance issue, mention the improvement and how the measurement was done (for educational purposes).
1. Do not leave comments unresolved. If PR comments have been addressed without making the requested code changes,
   explicitly mark them resolved with an appropriate comment explaining why you're resolving it. If you intend to resolve it
   in a follow-up PR, create a task and mention why this comment cannot be fixed in this PR. Leaving comments unresolved
   sets a wrong precedent for other contributors that it's ok to ignore comments.
1. In the interest of time, discuss the PR/comments in person if it's difficult to explain in writing. Document the
   resolution in the PR for the educational benefit of others. Don't just mark the comment resolved saying 'based on offline
   discussion'.
1. Add comments, if not obvious, in the PR to help the reviewer navigate your PR faster. If this is a big change, include
   a short design doc (docs/ folder).
1. Unit tests are mandatory for all PRs (except when the proposed changes are already covered by existing unit tests).
1. Do not use PRs as scratch pads for development as they consume valuable build/CI cycles for every commit. Build and
   test your changes for at least one environment (Windows/Linux/Mac) before creating a PR.
1. Keep it small. If the feature is big, it's best to split into multiple PRs. Modulo cosmetic changes, a PR with more
   than 10 files is notoriously hard to review. Be kind to the reviewers.
1. Separate cosmetic changes from functional changes by making them separate PRs.
1. The PR author is responsible for merging the changes once they're approved.
1. If you co-author a PR, seek review from someone else. Do not self-approve PRs.

## GitHub flow Workflow

Please follow [GitHub Flow](https://githubflow.github.io/)

In the GitHub flow workflow, there are 2 different branch types:

- Main: contain production-ready code that can be released.
- Feature: develop new features for the upcoming releases, to be merged to master after review

## Commit message

We try to follow [Conventional Commits](https://www.conventionalcommits.org) for commit messages and PR titles
through [gitlint](https://jorisroovers.com/gitlint/).

You then need to install the pre-commit hook like so:

```
pre-commit install --hook-type commit-msg
```

## More details:

- [4 branching workflows for Git](https://medium.com/@patrickporto/4-branching-workflows-for-git-30d0aaee7bf)
- [A Note About Git Commit Messages](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)
