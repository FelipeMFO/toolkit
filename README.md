# FM's Toolkit 

The objective of this repository is to be my tool case for data projects.

- Containerization or virtual environments available.
- Linter used: flake8 https://code.visualstudio.com/docs/python/linting
- Docstring type: Google style guide https://google.github.io/styleguide/pyguide.html#doc-function-args
- Two branches of work:
    - _main_ -> basic project to be cloned and used.
    - _legacy_ -> project with all interesting functions I've gathered during my life.
- For further clues, Coda coding page: https://coda.io/d/Coding_d39U-xSSVWG/Python_suKBo#_lur8i

## Some patterns
- Modules -> PascalCase.
- Sources, notebooks, folders and data -> snake_case.
- Bashs, configurations, related -> kebab-case.
- Variables:
    - Singular objects -> 4 letters.
    - Plural objects -> 4 letters + 's'.
    - Methods and functions -> 3 or 4 letters (get, gen, set - load, dump, save).
- JSON:
    - Alphabetically.
    - Nested dicts below.
    - If a word gets too repeated in more objects, put this word as first and declare it.
- Jupyter notebooks:
    - Sections and subsections markdowns as nouns, not verbs.

## Structure

- __.project-name__: Folder of virtual environment of the project.
- __.vscode__: Configurations of the VSCode.
- __auxiliar_files__: Third part codes that might help the project.
- __config__: Folder containing:
    - __Variables.py__: Module responsible for loading the variables on the JSON in real time as objects of a declared object. Refreshing in real time.
    - __variables-for-script.json__: Variables of the projects declared as a JSON file considering the relative path from the root of the project, where scripts are situated.
    - __variables.json__: Variables of the projects declared as a JSON file considering the relative path from the `src` and `notebooks` folders.
- __data__: 
    - __raw__
    - __structured__: Processed data that will still be used inside the project.
    - __enriched__: Output data that will not be used inside the project.
- __documents__
- __drafts__: Code that was developed during the project and might be deprecated but still there as a test, backup or draft.
- __figures__: Figures created from notebooks or scripts.
- __h2o__: Specific folder for the AutoML framework used by me.
- __images__: Images that might help the project or be used inside the notebooks' markdowns.
- __models__
- __notebooks__
    - __exploration__: Exploratory data analysis and notebooks to test data generation.
    - __modeling__: Model development using data created and information raised in exploration analysis.
    - __evaluation__: Applying evaluations based on metrics or plots.
- __src__: Source code containing modules and their helpers (if necessary) divided by folder.
    - __feature_engineering__: Methods that creates features and combine different process to generate strucuted data.
    - __metrics__: Methods to evaluate models and assess data' hypothesis.
    - __modeling__: Methods used to develop models. Callbacks and seeds.
        - __nn__: Neural Networks structures.
    - __processing__: Methods to process data, filter raw data and recombine. The difference to feature engineering lies on the ideas added, if it's based on only filtering with thresholds, masks, etc... it's processing, if it combines different datasets, gets means, medians, etc... it's feature engineering.
    - __queries__: Normally SQL code.
    - __visualization__: Folder containing plots and visualization modules.
    - __DataDumper.py__
    - __DataLoader.py__
    - __MongoLoader.py__: Mongo loader module.
    - __utils.py__: General helper.
- __bs-script-gen-folders.sh__: Shell script to generate structure of the project.
- __Dockerfile__
- __python-scripts.py__: General python scripts.


---

# README template (Project name)

Briefly description about the project motivation and business or technical problem we are trying to solve with this.

## Project's objective

## Maintainability information
There are future works? Wich ones are ideas, and which ones are stakeholders' demands?
There are references that someone that continues this work in the future must take into consideration?

## Setup information

Follow the guidelines described at the Repository's Wiki to setup local environment, then run the following commands.

### Setting virtual env

```
$ python3 -m env .name-of-the-project

# Linux
$ source .name-of-the-project/bin/activate

# Windows
$ .\.name-of-the-project\Scripts\activate
```

### Build Image

```bash
$ sudo docker build --tag=template-project:1.0 .
```

#### Run the container

Set the ENV variables GLOBAL_VARIABLE_NAME in the machine and then run the following commands:

```bash
$ docker create -it -p 8000:8000 \
    -v $PWD:/home/user \
    -e GLOBAL_VARIABLE_NAME=/path/to/file \
    -v $GLOBAL_VARIABLE_NAME:/path/to/file:ro \
    --name template-project template-project:1.0

$ sudo docker start template-project
$ sudo docker exec -it template-project bash
$ jupyter lab --ip 0.0.0.0 --port 8000 --allow-root
```

**Further stuff instalation**

- googledrive.com
- dropbox.com
