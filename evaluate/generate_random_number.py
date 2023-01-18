import pickle
import random

output_path = "data_utils/random_numbers.pt"

number_of_rnd = 1000000

number_list = list( range(number_of_rnd) )
random.shuffle(number_list)

with open(output_path,'wb') as f_out:
    pickle.dump(number_list, f_out)
