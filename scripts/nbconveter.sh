#!/usr/bin/env bash
#
# Automatically exclude notebook outputs.
# https://medium.com/somosfit/version-control-on-jupyter-notebooks-6b67a0cf12a3
# https://pypi.org/project/nbstripout
# https://github.com/jupyter/nbdime
# https://github.com/mwouts/jupytext
# https://www.fotonixx.com/posts/data-science-vcs/
# https://departmentfortransport.github.io/ds-processes/Coding_standards/ipython.html
# https://zhauniarovich.com/post/2020/2020-06-clearing-jupyter-output/

if command -v jupyter >/dev/null  2>&1; then
    echo "Clean Outputs Cells and converting notebooks to scripts ..."
    # https://gist.github.com/tylerneylon/697065ca5906c185ec6dd3093b237164
    # Convert all new Jupyter notebooks to straight Python files for easier code reviews.
    for file in $(git diff --cached --name-only); do
        if [[ $file == *.ipynb ]]; then
            echo -e "Converting ${file} ..."
            jupyter nbconvert --ClearOutputPreprocessor.enabled=True --clear-output --inplace "${file}"
            jupyter nbconvert --to script "${file}"
            git add "${file%.*}".py
        fi
    done
else
    echo "Jupyter Notebook not installed; Please install Jupyter Notebook."
    exit 1
fi

# EOF
