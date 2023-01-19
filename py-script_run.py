from config.Variables import VariablesScript
from src.processing.Processing import Processing
from src.DataLoader import DataLoader

ITERABLE = (1, 2, 3)

if __name__ == "__main__":
    varv = VariablesScript()
    load_sr = DataLoader(varv.PATH_DATA_RAW)
    proc = Processing()

    for source in ITERABLE:
        pass
