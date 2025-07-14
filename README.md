# di-python-template
Base python template

# Things to include in the `README.md` file
## Project description
    Decribe the project repo.
## Project Structure
## Setup instructions
    Help new team members get up to speed with the project. 
## Usage and examples [if applicable]
## Code Quality

This repository has strict linting and typing checks to ensure consistency across the codebase.
These checks are run in the CI/CD pipeline, and they are a requirement for merging a PR.
To run these checks locally, you can use the following commands:

```bash
# Linting
ruff check --output-format=concise --config ruff.toml .
# Typing
mypy --ignore-missing-imports --install-types --non-interactive .
```

Alternatively, you can install `act` and run the GitHub Actions locally:

- Docker must be installed and running.

```bash
brew install act
act -j linting-and-typing
```




---
---
# After creating the git repo
## Setting up the `Ruleset`
- Rulesets define whether collaborators can delete or force push and set requirements for any pushes, such as passing status checks or a linear commit history.

    ### Steps: 
    - Go to `Settings` -> `Rules` -> `Rulesets`:
        - Set the `Ruleset Name`: 
            Example:
            ```bash
                review-required
            ```
        - Set `Enforcement status` to `Active`
        - Set `Target branches` -> `Branching targeting criteria` -> `Add target` to `default branch`
        - Under `Branch rules`, tick 
            - Restrict deletion
            - Require a pull request before merging - `Required approvals` to 1 (or more)
            - Block force pushes
        - Click on `Create`

    - Under `Settings` -> `Collaboration and teams` : You can see everyone who has access to your repository and adjust permissions.



## Tips
- Use README.md to Documnet the Repository
    - The README.md file is the first thing a visitor sees when they land on your repository. It's a great place to provide a quick overview of the repository, its purpose, and how to get started with it.
- Secure your repository with branch protection rules
- Utilise .gitignore
- Maintain a Clean Commit History


### PR & Reviews
PR-creator must invite reviewers and also ask for review in slack with git url to the PR when the PR is ready for review.

### Merging
Every merge to `master`( or in case of `dn-umbrella` `dev` and/or `master`) should go through a review PR and should be merged by PR-creator.


### Some useful readings:
- [Python guide](https://gist.github.com/ruimaranhao/4e18cbe3dad6f68040c32ed6709090a3) by Ruimaranhao
- To make readme: [makereadme.com](https://www.makeareadme.com/)