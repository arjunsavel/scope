Integrations
----------------
As a growing open-source project, :code:`scope` has made use of a number of freely
available integrations and services, which are summarized below:

Code style
------------
With a consistent style, bugs are easier to find, code is easier to maintain, and enhancements are easier to make.

- `black <https://black.readthedocs.io/en/stable/>`_: a Python code-formatter.
  It generally conforms to `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_
  standards, but it does so in a very deterministic manner, which is helpful for
  our code. The basic pre-commit configuration is location in :code:`.pre-commit-config.yaml`,
  while more detailed configuration is detailed in :code:`pyproject.toml`. black is also
  implemented as a GitHub action, with the relevant configuration in
  :code:`.github/workflows/style`. To apply the same methodology to our Jupyter
  notebooks, we also make use of
  `black_nbconvert <https://github.com/dfm/black_nbconvert>`_.

- `isort <https://isort.readthedocs.io/en/latest/>`_: a tool for sorting imports
  in Python files. As with black, the basic pre-commit configuration is location
  in :code:`.pre-commit-config.yaml`, while more detailed configuration is detailed in
  :code:`pyproject.toml`. black is also implemented as a GitHub action, with the
  relevant configuration in :code:`.github/workflows/style`.

- `yamllint <https://github.com/adrienverge/yamllint>`_: lints our YAML files,
  checking them for bugs, syntax errors, and general style. The associated
  configuration file is :code:`.yamllint`. Implemented as a GitHub check and pre-commit
  hook.

Continuous integration
-----------------------
With continuous integration, our testing and releasing processes are automated, allowing for
minimization of bugs and ultimately improvement of user experience.


- `Codecov <https://codecov.io/gh>`_: checks what percentage of our code base
  is covered in our automated tests. Ideally, we'd like to keep our coverage above
  95% if it is in a production environment. Relevant configurations are noted in
  :code:`codecov.yml`.

- `Dependabot <https://dependabot.com/>`_: automatically checks whether
  dependencies are kept up-to-date. Configuration is held in :code:`.github/dependabot.yml`.

- `GitHub Actions <https://github.com/features/actions>`_: A number of the
  services described in this page are implemented as GitHub Actions, which manifest as
  `checks <https://developer.github.com/v3/checks/>`_ on different commits. In
  general, we would like all of our checks to pass before pull requests are
  merged. Configuration files for GitHub Actions are contained in
  :code:`.github/workflows`.

- `pre-commit hooks <https://pre-commit.com/>`_: these can be applied to any
  local commits. The associated configuration file is :code:`.pre-commit-config.yml`.
