from fastai.tabular.all import *
from fastai.collab import *
from sklearn.metrics import r2_score
import pandas as pd
import csv

# Load data
df = pd.read_csv('D:/RecommenderSystem/bound_value.csv')

# Define categorical and continuous variables
cat_names = ['dish_name']
cont_names = ['Calories', 'Carbs', 'Fat', 'Protein', 'Sodium', 'Sugar', 'Fiber', 'Potass.', 'Iron', 'Calcium',
              'Sat Fat', 'Chol', 'Vit A', 'Vit C', 'Trn Fat']

# Normalize continuous variables
procs = [Categorify, Normalize]

# Split data into training and validation sets
splits = RandomSplitter(valid_pct=0.2)(range_of(df))

# Create dataloaders
to = TabularPandas(df, procs, cat_names=cat_names, cont_names=cont_names, y_names='Calories', splits=splits)
dls = to.dataloaders(bs=64)


# Define the model
class NutritionalDotProduct(Module):
    def __init__(self, n_dishes, n_factors):
        self.dish_factors = Embedding(n_dishes, n_factors)
        self.nutrition_factors = nn.Linear(len(cont_names), n_factors)
        self.bias = nn.Parameter(torch.zeros((n_dishes,)))

    def forward(self, x_cat, x_cont):
        dish_f = self.dish_factors(x_cat[:, 0])
        nutrition_f = self.nutrition_factors(x_cont)
        dot_product = (dish_f * nutrition_f).sum(dim=1)
        return dot_product + self.bias[x_cat[:, 0]]


# Create the learner
n_factors = 5

model = NutritionalDotProduct(len(to.classes['dish_name']), n_factors)
metrics = [r2_score]
learn = Learner(dls, model, loss_func=MSELossFlat(), metrics=metrics)

# Train the model
learn.fit_one_cycle(5, 5e-3)

# Get the embeddings
dish_w = learn.model.dish_factors.weight.detach().cpu()
nutrition_w = learn.model.nutrition_factors.weight.detach().cpu()

# Perform PCA on the nutrition factors to visualize them
nutrition_pca = nutrition_w.pca(3)
fac0, fac1, fac2 = nutrition_pca.t()
print(learn.validate())
# Create test dataloader
test_dl = dls.test_dl(df.iloc[:50])

# Test the model
result = learn.validate(dl=test_dl)
print(result)


# Nutrition Function
# First Function


def get_nutrition_infos(dish_name):
    with open('D:/RecommenderSystem/bound_value.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['dish_name'] == dish_name:
                nutrition = {
                    "Calories": row['Calories'],
                    "Carbohydrates": row['Carbs'],
                    "Protein": row['Protein'],
                    "Fat": row['Fat'],
                    "Sodium": row['Sodium'],
                    "Sugar": row['Sugar']
                }
                return nutrition
        # If the dish name is not found in the CSV file, return None
        return None


# Extract the learned embeddings and biases for dishes


dish_factors = learn.model.dish_factors
dish_bias = learn.model.bias


# Get the indices of the dishes with the highest and lowest biases


def get_dish_info():
    idxs_low_bias = dish_bias.argsort()[:50]
    # Get the names of the dishes with the lowest biases
    # Create response data
    response_data = {
        'Default Recommendations': []
    }
    dish_names_low_bias = [dls.classes['dish_name'][i] for i in idxs_low_bias]
    dish_nutrition_low_bias = [get_nutrition_infos(dish_name) for dish_name in dish_names_low_bias]
    for name, nutrition in zip(dish_names_low_bias, dish_nutrition_low_bias):
        response_data['dishes_with_lowest_biases'].append({'name': name, 'nutrition': nutrition})

    return response_data


# Recommendations on user Input Dish


def get_dish_input(dish):
    # Check if the entered dish exists in the dataset
    if dish in dls.classes['dish_name']:
        # Get index of entered dish
        idx = dls.classes['dish_name'].o2i[dish]

        # Get dish names sorted by bias
        idxs_low_bias = (dish_factors.weight.data[idx] - dish_factors.weight.data).pow(2).sum(dim=1).argsort()[:10]

        # Create response data
        response_data = {
            'dishes_with_lowest_biases': []
        }

        # Add dish names and nutrition to the response data
        dish_names_low_bias = [dls.classes['dish_name'][i] for i in idxs_low_bias]
        dish_nutrition_low_bias = [get_nutrition_infos(dish_name) for dish_name in dish_names_low_bias]
        for name, nutrition in zip(dish_names_low_bias, dish_nutrition_low_bias):
            response_data['dishes_with_lowest_biases'].append({'name': name, 'nutrition': nutrition})

        return response_data
    else:
        return {'error': 'Entered dish not found in dataset.'}
