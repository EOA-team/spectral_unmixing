from models.NN import SimpleNN
from models.RF import RFRegressor
from models.SVR import SVRegressor

MODELS = {
    "NN": SimpleNN,
    "RF": RFRegressor,
    "SVR": SVRegressor
}
