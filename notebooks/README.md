# Jupyter Notebooks

Notebook packages like [Jupyter notebook](http://jupyter.org/) are effective tools for exploratory data analysis,
fast prototyping, and communicating results; however, between prototyping and communicating results, **code should be
factored out into proper python modules**.

All jupyter notebooks should go into the `notebooks` subfolder of the respective experiment.
To make best use of the folder structure, the parent folder of each notebook should be added to the import path.

This allows to then import the helper and the script module belonging to a specific experiment as follows:

```
import imports
# or
from imports import package_installed
```

You should also enable `nbstripout`, so that only clean versions of your notebooks get committed to git.
Use `pre-commit` and `nbstripout` to remove bulky notebook output data before committing changes.

It may be necessary or useful to keep certain output cells of a Jupyter notebook, for example charts or graphs visualising
some set of data. To do this, [according to the documentation for the `nbstripout` package][nbstripout], either:

1. add a `keep_output` tag to the desired cell; or
1. add `"keep_output": true` to the desired cell's metadata.

You can access cell tags or metadata in Jupyter by enabling the "Tags" or
"Edit Metadata" toolbar (View > Cell Toolbar > Tags; View > Cell Toolbar >
Edit Metadata).

For the tags approach, enter `keep_output` in the text field for each desired cell, and
press the "Add tag" button. For the metadata approach, press the "Edit Metadata" button
on each desired cell, and edit the metadata to look like this:

```json
{
  "keep_output": true
}
```

This will tell the hook not to strip the resulting output of the desired cell(s), allowing the output(s) to be committed.

The `.envrc` file should automatically add the entire project path into the `PYTHONPATH` environment variable.
This should allow you to directly import `src/time_series_metrics` in your notebook.

### Tips

1. Add the following to your notebook (or IPython REPL):

```
%load_ext autoreload
%autoreload 2
```

Now when you save code in a python module, the notebook will automatically load in the latest changes without you having to
restart the kernel, re-import the module etc.

2. Don't install `jupyter`/`jupyterlab` in your environment, use `ipykernel`

You should avoid `jupyter`/`jupyterlab` as a dependency in the project environment.

Instead, add `ipykernel` as a dependency. This is a lightweight dependency that allows `jupyter`/`jupyterlab` installed elsewhere
(e.g. your main conda environment or system installation) to run the code in your project.

Run `python -m ipykernel install --user --name="pymdma"` from within your project environment to allow jupyter
to use your project's virtual environment. Afterwards a new kernel called time_series_metrics should be available in the
jupyter lab/jupyter notebook interface. Use that kernel for all notebooks related to this project.

The advantages of this are:

- You only have to configure `jupyter`/`jupyterlab` once
- You will save disk-space
- Faster install
- Colleagues using other editors don't have to install heavy dependencies they don't use (you wouldn't be happy if someone sent you
  code that depended on VScode/Pycharm/Spyder)

Note: `ipykernel` is also listed in `requirements/requirements_notebook.txt` so you do not need to add it.

[nbstripout]: https://github.com/kynan/nbstripout
