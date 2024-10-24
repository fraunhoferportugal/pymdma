#################################################################################
# GLOBALS                                                                       #
#################################################################################
include Makefile.include
#################################################################################
# Environment                                                                   #
#################################################################################
include Makefile.envs

#################################################################################
# Text Helpers                                                               	#
#################################################################################
+TERMINATOR := \033[0m
+WARNING := \033[1;33m [WARNING]:
+INFO := \033[1;33m [INFO]:
+HINT := \033[3;33m
+SUCCESS := \033[1;32m [SUCCESS]:

#################################################################################
# Conda                                                                   		#
#################################################################################
ifeq (,$(CONDA_EXE))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# Helper Functions																#
#################################################################################
# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
# from: https://stackoverflow.com/questions/10858261/how-to-abort-makefile-if-variable-not-set
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
        $(error Undefined $1$(if $2, ($2))$(if $(value @), \
                required by target `$@')))



#################################################################################
# Virtual Environments															#
#################################################################################
.PHONY: setup-docs setup-notebooks setup-all  \
		setup-image setup-tabular setup-time_series setup-text setup-prod \
		setup-dev-dep setup-all

## Build documentation virtual environment and install dependencies
setup-docs:
	@echo -e "$(INFO) Creating documentation virtual environment...$(TERMINATOR)" && \
	python3 -m venv .venv-docs && \
	source .venv-docs/bin/activate && \
	poetry run pip install --upgrade pip setuptools && \
	poetry install --all-extras --with dev,docs && \
	echo -e "$(SUCCESS) Virtual environment created successfully!$(TERMINATOR)" && \
	echo -e "$(HINT) Activate the virtual environment with: source .venv-docs/bin/activate$(TERMINATOR)"

## Build notebooks virtual environment and install dependencies
setup-notebooks: .venv-note/
	@echo -e "$(INFO) Creating notebooks virtual environment...$(TERMINATOR)" && \
	python3 -m venv .venv-note && \
	source .venv-note/bin/activate && \
	poetry run pip install --upgrade pip setuptools && \
	poetry install && \
	poetry install --with notebook && \
	echo -e "$(SUCCESS) Virtual environment created successfully!$(TERMINATOR)" && \
	echo -e "$(HINT) Activate the virtual environment with: source .venv-note/bin/activate$(TERMINATOR)"

## Build virtualenv for image data metrics development
setup-image:
	@echo -e "$(INFO) Creating image virtual environment...$(TERMINATOR)" && \
	python3 -m venv .venv-image && \
	source .venv-image/bin/activate && \
	poetry run pip install --upgrade pip setuptools && \
	poetry install --extras "image" && \
	echo -e "$(SUCCESS) Virtual environment created successfully!$(TERMINATOR)" && \
	echo -e "$(HINT) Activate the virtual environment with: source .venv-image/bin/activate$(TERMINATOR)"

## Build virtualenv for tabular data metrics development
setup-tabular:
	@echo -e "$(INFO) Creating tabular virtual environment...$(TERMINATOR)" && \
	python3 -m venv .venv-tabular && \
	source .venv-tabular/bin/activate && \
	poetry run pip install --upgrade pip setuptools && \
	poetry install --extras "tabular" && \
	echo -e "$(SUCCESS) Virtual environment created successfully!$(TERMINATOR)" && \
	echo -e "$(HINT) Activate the virtual environment with: source .venv-tabular/bin/activate$(TERMINATOR)"

## Build virtualenv for time series data metrics development
setup-time_series:
	@echo -e "$(INFO) Creating time_series virtual environment...$(TERMINATOR)" && \
	python3 -m venv .venv-time_series && \
	source .venv-time_series/bin/activate && \
	poetry run pip install --upgrade pip setuptools && \
	poetry install --extras "time_series" && \
	echo -e "$(SUCCESS) Virtual environment created successfully!$(TERMINATOR)" && \
	echo -e "$(HINT) Activate the virtual environment with: source .venv-time_series/bin/activate$(TERMINATOR)"

## Build virtualenv for text data metrics development
setup-text:
	@echo -e "$(INFO) Creating text virtual environment...$(TERMINATOR)" && \
	python3 -m venv .venv-text && \
	source .venv-text/bin/activate && \
	poetry run pip install --upgrade pip setuptools && \
	poetry install --extras "text" && \
	echo -e "$(SUCCESS) Virtual environment created successfully!$(TERMINATOR)" && \
	echo -e "$(HINT) Activate the virtual environment with: source .venv-text/bin/activate$(TERMINATOR)"

## Build all virtual environments (production)
setup-prod: # TODO - torch cpu install
	@echo -e "$(INFO) Creating production virtual environment...$(TERMINATOR)" && \
	python3 -m venv .venv-prod && \
	source .venv-prod/bin/activate && \
	poetry run pip install --upgrade pip setuptools && \
	poetry install --with prod --all-extras && \
	echo -e "$(SUCCESS) Virtual environment created successfully!$(TERMINATOR)" && \
	echo -e "$(HINT) Activate the virtual environment with: source .venv-prod/bin/activate$(TERMINATOR)"

## Build all virtual environments (development)
setup-all:
	@echo -e "$(INFO) Creating development virtual environment...$(TERMINATOR)" && \
	python3 -m venv .venv-dev && \
	source .venv-dev/bin/activate && \
	poetry run pip install --upgrade pip setuptools && \
	poetry install --with dev --all-extras && \
	echo -e "$(SUCCESS) Virtual environment created successfully!$(TERMINATOR)" && \
	echo -e "$(HINT) Activate the virtual environment with: source .venv-dev/bin/activate$(TERMINATOR)"


#################################################################################
# Update Dependencies															#
#################################################################################
.PHONY: update-requirements update-requirements-dev update-requirements-docs update-requirements-notebooks update-requirements-dempy update-requirements-all\
		update-requirements-image update-requirements-tabular update-requirements-time_series update-requirements-text

## Update project main requirements
# update-requirements:
# 	@echo -e "$(INFO) Updating requirements file...$(TERMINATOR)" && \
# 	poetry export --only main --without-hashes -f requirements.txt -o requirements/requirements.txt && \
# 	poetry export --only prod --without-hashes -f requirements.txt -o requirements/requirements-prod.txt && \
# 	echo -e "$(SUCCESS) Requirements updated successfully!$(TERMINATOR)"

# ## Update project development requirements
# update-requirements-dev:
# 	@echo -e "$(INFO) Updating development requirements file...$(TERMINATOR)" && \
# 	poetry export --only dev --without-hashes -f requirements.txt -o requirements/requirements-dev.txt && \
# 	echo -e "$(SUCCESS) Requirements updated successfully!$(TERMINATOR)"

# ## Update project documentation requirements
# update-requirements-docs:
# 	@echo -e "$(INFO) Updating documentation requirements file...$(TERMINATOR)" && \
# 	poetry export --only docs --without-hashes -f requirements.txt -o requirements/requirements-docs.txt && \
# 	echo -e "$(SUCCESS) Requirements updated successfully!$(TERMINATOR)"

# ## Update project notebooks requirements
# update-requirements-notebooks:
# 	@echo -e "$(INFO) Updating notebooks requirements file...$(TERMINATOR)" && \
# 	poetry export --only notebook --without-hashes -f requirements.txt -o requirements/requirements-notebook.txt && \
# 	echo -e "$(SUCCESS) Requirements updated successfully!$(TERMINATOR)"


# ## Update image requirements
# update-requirements-image:
# 	@echo -e "$(INFO) Updating image requirements file...$(TERMINATOR)" && \
# 	poetry export --only image --without-hashes -f requirements.txt -o requirements/requirements-image.txt && \
# 	echo -e "$(SUCCESS) Requirements updated successfully!$(TERMINATOR)"


# ## Update time_series requirements
# update-requirements-time_series:
# 	@echo -e "$(INFO) Updating time_series requirements file...$(TERMINATOR)" && \
# 	poetry export --only time_series --without-hashes -f requirements.txt -o requirements/requirements-time_series.txt && \
# 	echo -e "$(SUCCESS) Requirements updated successfully!$(TERMINATOR)"

# ## Update tabular requirements
# update-requirements-tabular:
# 	@echo -e "$(INFO) Updating tabular requirements file...$(TERMINATOR)" && \
# 	poetry export --only tabular --without-hashes -f requirements.txt -o requirements/requirements-tabular.txt && \
# 	echo -e "$(SUCCESS) Requirements updated successfully!$(TERMINATOR)"

# ## Update text requirements
# update-requirements-text:
# 	@echo -e "$(INFO) Updating text requirements file...$(TERMINATOR)" && \
# 	poetry export --only text --without-hashes -f requirements.txt -o requirements/requirements-text.txt && \
# 	echo -e "$(SUCCESS) Requirements updated successfully!$(TERMINATOR)"

update-requirements-prod:
	@echo -e "$(INFO) Updating prod requirements file...$(TERMINATOR)" && \
	poetry export --all-extras --with prod --without-hashes -f requirements.txt -o requirements/requirements-prod.txt && \
	echo -e "$(SUCCESS) Requirements updated successfully!$(TERMINATOR)"

# ## Update all project requirements
# update-requirements-all: update-requirements update-requirements-dev update-requirements-docs update-requirements-notebooks update-requirements-dempy \
# 		update-requirements-image update-requirements-tabular update-requirements-time_series update-requirements-text


#################################################################################
# Dev dependencies																#
#################################################################################
.PHONY: install-dev-tools uninstall-pre-commit install-dev-all install-pre-commit


## Install only dev dependencies in the current environment
install-dev-tools:
	@:$(call check_defined, VIRTUAL_ENV, Please activate a virtual environment before running this command.)
	@echo -e "$(INFO) Installing development dependencies...$(TERMINATOR)" && \
	poetry install --only dev && \
	echo -e "$(SUCCESS) Development tools installed successfully!$(TERMINATOR)"

## Install pre-commit hooks in the current environment
install-pre-commit:
	@:$(call check_defined, VIRTUAL_ENV, Please activate a virtual environment before running this command.)
	@echo -e "$(INFO) Setting up pre-commit hooks...$(TERMINATOR)" && \
	poetry run pre-commit install --install-hooks -t pre-commit -t commit-msg && \
	poetry run pre-commit autoupdate && \
	echo -e "$(SUCCESS) Setup complete!$(TERMINATOR)"

## Install development dependencies and pre-commit hooks in the current environment
install-dev-all: install-dev-tools install-pre-commit
	@echo -e "$(SUCCESS) Development tools and pre-commit installed successfully!$(TERMINATOR)"

## Remove pre-commit hooks
uninstall-pre-commit:
	@:$(call check_defined, VIRTUAL_ENV, Please activate a virtual environment before running this command.)
	@echo -e "$(INFO) Removing pre-commit hooks...$(TERMINATOR)" && \
	poetry run pre-commit uninstall && \
	poetry run pre-commit uninstall --hook-type pre-push && \
	rm -rf .git/hooks/pre-commit && \
	echo -e "$(SUCCESS) Pre-commit hooks removed!$(TERMINATOR)"


#################################################################################
# Cleanup																		#
#################################################################################
.PHONY: clean clean-test clean-hydra clean-all

## Remove build and Python artifacts
clean:
	rm -rf .venv*
	find . -type f -name '*.pyc' -delete
	find . -type f -name "*.DS_Store" -ls -delete
	find . -type f -name '*~' -exec rm -f {} +
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

## Remove test, coverage artifacts
clean-test:
	rm -fr .tox/
	rm -fr .nox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache

## Remove hydra outputs
clean-hydra:
	rm -rf outputs
	rm -rf runs
	rm -rf multirun
	rm -rf mlruns

## Remove all artifacts
clean-all: clean clean-test clean-hydra

# TODO
#################################################################################
# Testing																		#
#################################################################################
.PHONY: test format-check format-fix coverage coverage-html lint check-safety

## Run Pytest tests
test:
	@echo -e "$(INFO) Running tests...$(TERMINATOR)" && \
	source .venv-dev/bin/activate && \
	poetry run pytest -vvv tests/

## Verify formatting style
format-check: setup-all
	@echo -e "$(INFO) Checking code formatting...$(TERMINATOR)" && \
	source .venv-dev/bin/activate && \
	poetry run isort --check-only src/ && \
	poetry run black --check src/ && \

## Fix formatting style. This updates files
format-fix: setup-all
	@echo -e "$(INFO) Fixing code formatting...$(TERMINATOR)" && \
	source .venv-dev/bin/activate && \
	poetry run isort src/ && \
	poetry run black src/ && \

## Generate test coverage reports
coverage: setup-all
	@echo -e "$(INFO) Running coverage...$(TERMINATOR)" && \
	source .venv-dev/bin/activate && \
	poetry run pytest --cov=src/pymdma tests/ && \
	poetry run coverage report

## Generate HTML coverage report
coverage-html: coverage
	@echo -e "$(INFO) Generating HTML coverage report...$(TERMINATOR)" && \
	source .venv-dev/bin/activate && \
	poetry run coverage html

## Check code for lint errors
lint: setup-all
	@echo -e "$(INFO) Running linters...$(TERMINATOR)" && \
	source .venv-dev/bin/activate && \
	poetry run flake8 src/ && \
	poetry run mypy src/

## Check for package vulnerabilities
check-safety: setup-all
	@echo -e "$(INFO) Checking dependencies for security vulnerabilities...$(TERMINATOR)" && \
	source .venv-dev/bin/activate && \
	poetry run safety check -r requirements/requirements.txt && \
	poetry run safety check -r requirements/requirements-prod.txt && \
	poetry run bandit -ll --recursive hooks


#################################################################################
# Version Control																#
#################################################################################
.PHONY: commit push bump bump_major bump_minor bump_micro jenkins_bump changelog change-version release dvc-download dvc-upload push-all

## Commit using Conventional Commit with Commitizen
commit:
	ifndef VIRTUAL_ENV
		error "No virtual environment detected. Please activate a virtual environment before running this command."
	else
		@echo -e "$(INFO) Committing changes with Commitizen...$(TERMINATOR)" && \
		poetry run cz commit
	endif

## Git push code and tags
push:
	@echo -e "$(INFO) Pushing changes to remote...$(TERMINATOR)" && \
	git push && \
	git push --tags

## Bump semantic version based on the git log
bump:
	ifndef VIRTUAL_ENV
			error "No virtual environment detected. Please activate a virtual environment before running this command."
	else
		@echo -e "$(INFO) Bumping version...$(TERMINATOR)" && \
		poetry run cz bump
	endif

## Bump to next major version (e.g. X.Y.Z -> X+1.Y.Z)
bump_major:
	echo "$(CURRENT_VERSION_MAJOR)" > VERSION

## Bump to next minor version (e.g. Y.X.Y -> Y.X+1.Y)
bump_minor:
	echo "$(CURRENT_VERSION_MINOR)" > VERSION

## Bump to next micro version (e.g. Y.Y.X -> Y.Y.X+1)
bump_micro:
	echo "$(CURRENT_VERSION_MICRO)" > VERSION

## Jenkins version bump
jenkins_bump:
	@git config --global user.email "-"
	@git config --global user.name "Jenkins"
	@(git tag --sort=-creatordate | grep -E '^\d+\.\d+\.\d+$$' || echo '0.0.0') | head -n 1 > VERSION
	@/scripts/bump $$(git log -1 --pretty=%B)
	@git tag $$(cat VERSION)
	@git push origin $$(cat VERSION)

## Generate changelog
changelog:
	ifndef VIRTUAL_ENV
			error "No virtual environment detected. Please activate a virtual environment before running this command."
	else
		@echo -e "$(INFO) Generating changelog...$(TERMINATOR)" && \
		poetry run cz changelog

## Release new version.
release:
	git commit -am "bump: Release code version $(VERSIONFILE)"
	git tag -a v$(VERSIONFILE) -m "bump: Release tag for version $(VERSIONFILE)"
	git push
	git push --tags

## Change to previous model and data version (e.g. make change-version v="0.1.0")
change-version:
	git checkout v$v && \
	dvc checkout

## Get data from DVC remote
dvc-download:
	dvc pull

## Push data to DVC remote
dvc-upload:
	dvc push

## Push code and data
push-all: push dvc-upload


#################################################################################
# Documentation																	#
#################################################################################
.PHONY: mkdocs-build mkdocs-serve mkdocs-clean

## Generate MKDocs documentation
mkdocs-build: setup-docs
	@echo -e "$(INFO) Building documentation...$(TERMINATOR)" && \
	source .venv-docs/bin/activate && \
	poetry run mkdocs build

## Serve MKDocs documentation on localhost:8000
mkdocs-serve: setup-docs
	@echo -e "$(INFO) Serving documentation...$(TERMINATOR)" && \
	source .venv-docs/bin/activate && \
	poetry run mkdocs serve

## Clean MKDocs documentation
mkdocs-clean:
	rm -rf site/


#################################################################################
# Help																			#
#################################################################################
.DEFAULT_GOAL := help

.PHONY: help

## Show this help message
help:
	@echo "$$(tput bold)              ** Available rules: ** $$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=25 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf " - %s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
