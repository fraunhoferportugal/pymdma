# Loading environment variables

[We use `python-dotenv` to load environment variables][python-dotenv], as these are only loaded when
inside the project folder. This can prevent accidental conflicts with identically named
variables. Alternatively you can use [`direnv` to load environment variables][direnv] if
you meet [certain conditions](#installing-direnv).

## Using `python-dotenv`

To load the environment variables, first make sure you have
python-dotenv install, and [make sure you have a `.secrets` file to store
secrets and credentials](#storing-secrets-and-credentials). Then to load in the
environment variables into a python script see instructions in `.env` file.

## Using `direnv`

To load the environment variables, first [follow the `direnv` installation
instructions](#installing-direnv), and [make sure you have a `.secrets` file to store
secrets and credentials](#storing-secrets-and-credentials). Then:

1. Open your terminal;
1. Install `direnv`. See instructions below.
1. Navigate to the project folder; and
   - You should see the following message:
     ```shell
     direnv: error .envrc is blocked. Run `direnv allow` to approve its content.
     ```
1. Allow `direnv`.
   ```shell
   direnv allow
   ```

You only need to do this once, and again each time `.envrc` and `.secrets` are modified.

### Installing `direnv`

1. Open your terminal;
1. Install [`direnv`](https://direnv.net/docs/installation.html);
1. Add the shell hooks to your `.bash_profile`;
   ```shell
   echo 'eval "$(direnv hook bash)"' >> ~/.bash_profile
   ```
1. Check that the shell hooks have been added correctly; and
   ```shell
   cat ~/.bash_profile
   ```
   - This should display `eval "$(direnv hook bash)"`
1. Restart your terminal.

## Storing secrets and credentials

Secrets and credentials must be stored in the `.secrets` file. This file is not
version-controlled, so no secrets should be committed to GitHub.

Open this new `.secrets` file using your preferred text editor, and add any secrets as
environmental variables. For example, to add a JSON credentials file:

```shell
APPLICATION_CREDENTIALS="path/to/credentials.json"
```

Once complete, make sure the `.secrets` file has the following line uncommented out:

```shell
source_env ".secrets"
```

This ensures [`direnv`][direnv] loads the `.secrets` file using `.envrc` without
version-controlling `.secrets`.

[direnv]: https://direnv.net/
[python-dotenv]: https://saurabh-kumar.com/python-dotenv/
