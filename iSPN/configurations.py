
standard_run = {
    "batch_size": 64,
    "num_examples_to_generate": 16
}

standard_configuration = [{
    "latent_dim": 50,
    "hidden1": 0,
    "filters1": 64,     # filters 1. conv upsampling
    "filters2": 32,     # filters 2. conv upsampling
    "name": ""
}]

good_configuration = standard_configuration[0].copy()
good_configuration["latent_dim"] = int(20)
good_configuration["name"] = "good"
# good_configuration["hidden1"] = 500
# good_configuration["filters1"] = 64
# good_configuration["filters2"] = 32
# Transform to arry
good_configuration = [good_configuration]


"""
Change size of latent space

Result:
Best configuration latent space = 20 (fastest learning)
10 is signifficantly worse, but performance does not really 
increase for higher latent spaces
"""
latent_space_tests = []
for i in range(1, 11):
    dic = standard_configuration[0].copy()
    dic["latent_dim"] = i * 10
    dic["name"] = str(i * 10) + "_"
    latent_space_tests.append(dic)


"""
Change number of hidden neurons in first layer

Result:
Best configuration hidden1 = 900 (more seems better)
It seems as more hidden layers increase the result
by a tiny amount. 10 hidden neurons are worse than no
layer, the rest is slightly better
"""
hidden1_test = []
for i in range(0, 10):
    dic = standard_configuration[0].copy()
    dic["hidden1"] = i * 100
    dic["name"] = str(i * 10) + "_"
    hidden1_test.append(dic)


"""
Change number of filters in upsampling layer 
Test all combinations: 7*7 different types
Result:
More filters is 
"""
filters_test = []
for i in range(0, 7):
    for j in range(0, 7):
        dic = standard_configuration[0].copy()
        dic["name"] = str(2**(i+3)) + str(2**(j+2))
        dic["filters1"] = 2**(i+3)
        dic["filters2"] = 2**(j+2)
        filters_test.append(dic)

double_config = [
    {"leaf_p1": None,
     "leaf_p2": 8,
     "leaf_p3": 16,
     "sum_p1": None,
     "sum_p2": 10},
    {"leaf_p1": None,
     "leaf_p2": 8,
     "leaf_p3": 16,
     "sum_p1": None,
     "sum_p2": 10}]

all_configs = [
    {"leaf_p1": None,
     "leaf_p2": 8,
     "leaf_p3": 16,
     "sum_p1": None,
     "sum_p2": 5},

    {"leaf_p1": None,
     "leaf_p2": 16,
     "leaf_p3": 32,
     "sum_p1": None,
     "sum_p2": 10},

    {"leaf_p1": None,
     "leaf_p2": 32,
     "leaf_p3": 64,
     "sum_p1": None,
     "sum_p2": 20},

    {"leaf_p1": None,
     "leaf_p2": 32,
     "leaf_p3": 16,
     "sum_p1": None,
     "sum_p2": 10},

    {"leaf_p1": None,
     "leaf_p2": 32,
     "leaf_p3": 16,
     "sum_p1": None,
     "sum_p2": 16},

    {"leaf_p1": None,
     "leaf_p2": 64,
     "leaf_p3": 64,
     "sum_p1": None,
     "sum_p2": 64},

    {"leaf_p1": None,
     "leaf_p2": 32,
     "leaf_p3": 32,
     "sum_p1": None,
     "sum_p2": 32}]
