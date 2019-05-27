from preprocess import get_data,get_permutations,read_csv
from type_1_model import build_model
from type_1_loader import DataGenerator,get_valid

path = '/Users/lvyufeng/Documents/captcha_train_set/type1_train/{}'
train_data,valid_data = read_csv(path.format('type1_train.csv'))

training_generator = DataGenerator(train_data,path)
x_valid,y_valid = get_valid(valid_data,path)
model = build_model()
model.summary()
model.fit_generator(training_generator, epochs=50,validation_data=(x_valid, y_valid),max_queue_size=10,workers=1)