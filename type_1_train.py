from preprocess import get_data,get_permutations,read_csv
from type_1_model import build_model
from type_1_loader import DataGenerator


train_datas = read_csv('/Users/lvyufeng/Documents/captcha_train_set/type1_train/type1_train.csv')

training_generator = DataGenerator(train_datas)
model = build_model()
model.summary()
model.fit_generator(training_generator, epochs=50,max_queue_size=10,workers=1)