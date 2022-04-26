from utils import Utils
from models import Models

if __name__ == "__main__":

    utils = Utils()
    models = Models()

    data = utils.load_from_csv('./data/raw/felicidad.csv')

    X,y = utils.features_target(data,['score','rank','country'], ['score'])

    models.gid_training(X,y)
    


