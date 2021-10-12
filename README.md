# FM's Toolkit 

The objective of this repository is to be my tool case/arsenal for data projects

- TODO: Colocar aqui os comandos de inicialização de git e de envinronment
- TODO: Colocar informações do lint (flake8) https://code.visualstudio.com/docs/python/linting

## Project name

Briefly description about the project motivation and business or technical problem we are trying to solve with this.

## Project's objective

## Maintainability information
There are future works? Wich ones are ideas, and which ones are stakeholders' demands?
There are references that someone that continues this work in the future must take into consideration?

## Setup information

Follow the guidelines described at the Repository's Wiki to setup local environment, then run the following commands.

### Build Image

```bash
$ sudo docker build --tag=template-project:1.0 .
```

### Run the container

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

**Do other people running this analysis need to install any custom package on their computer or create additional files?**

No