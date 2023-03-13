import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# 加载转录组数据
data = pd.read_csv('transcriptome_data.csv')

# 将数据分为训练集和测试集
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)

# 提取特征和标签
train_features = train_data.drop(['patient_id', 'recurrence'], axis=1)
train_labels = train_data['recurrence']
test_features = test_data.drop(['patient_id', 'recurrence'], axis=1)
test_labels = test_data['recurrence']

# 构建模型
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=train_features.shape[1]))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_features, train_labels, epochs=10, batch_size=32)

# 评估模型
test_loss, test_acc = model.evaluate(test_features, test_labels)
print('Test accuracy:', test_acc)
