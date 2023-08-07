import numpy as np

def best_4_lin_reg(seed=1489683273):
    np.random.seed(seed)
    x = np.random.random((100, 9)) * 200 - 100
    weights = np.array([1, 9, 13, 15, 11, 7, 13, 20, 12])
    y = np.dot(x, weights)
    return x, y


def best_4_dt(seed=1489683273):
    np.random.seed(seed)
    x = np.random.random((100, 6)) * 200 - 100
    y = (x[:, 0] > 50).astype(int)
    return x, y


def author():
    return "aishwary"


if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    print("The author of this file is 'aishwary'")
