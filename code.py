import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.integrate import simps
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from statsmodels.nonparametric.kernel_regression import KernelReg

class FunctionalKNN(BaseEstimator):
    """
    处理函数型数据、分类数据和混合协变量的非参数回归与分类
    """
    
    def __init__(self, k=5, functional_metric='l2', categorical_metric='hamming', 
                 continuous_metric='euclidean', weights=None):
        self.k = k
        self.functional_metric = functional_metric
        self.categorical_metric = categorical_metric
        self.continuous_metric = continuous_metric
        self.weights = weights
        
    def _functional_distance(self, X1, X2):
        """计算函数型数据距离"""
        if self.functional_metric == 'l2':
            # L2距离（积分平方差）
            return np.sqrt(simps((X1 - X2)**2, dx=self.dt))
        elif self.functional_metric == 'derivative':
            # 基于导数的距离
            deriv1 = np.gradient(X1, self.dt)
            deriv2 = np.gradient(X2, self.dt)
            return np.sqrt(simps((X1 - X2)**2 + (deriv1 - deriv2)**2, dx=self.dt))
        else:
            return pairwise_distances([X1], [X2], metric=self.functional_metric)[0,0]
    
    def _mixed_distance(self, row1, row2):
        """计算混合数据类型的总距离"""
        total_dist = 0
        
        # 函数型数据距离
        if self.functional_cols:
            func_dist = 0
            for col in self.functional_cols:
                func_dist += self._functional_distance(row1[col], row2[col])
            total_dist += func_dist * self.weights.get('functional', 1.0)
        
        # 连续数据距离
        if self.continuous_cols:
            cont_data1 = [row1[col] for col in self.continuous_cols]
            cont_data2 = [row2[col] for col in self.continuous_cols]
            cont_dist = pairwise_distances([cont_data1], [cont_data2], 
                                         metric=self.continuous_metric)[0,0]
            total_dist += cont_dist * self.weights.get('continuous', 1.0)
        
        # 分类数据距离
        if self.categorical_cols:
            cat_data1 = [row1[col] for col in self.categorical_cols]
            cat_data2 = [row2[col] for col in self.categorical_cols]
            cat_dist = sum(1 for a, b in zip(cat_data1, cat_data2) if a != b)
            total_dist += cat_dist * self.weights.get('categorical', 1.0)
            
        return total_dist
    
    def fit(self, X, y, functional_cols=None, continuous_cols=None, 
            categorical_cols=None, dt=0.1):
        """
        拟合模型
        
        Parameters:
        X: 包含混合类型数据的DataFrame
        y: 目标变量
        functional_cols: 函数型数据列名列表
        continuous_cols: 连续数据列名列表  
        categorical_cols: 分类数据列名列表
        dt: 函数型数据的采样间隔
        """
        self.X = X.copy()
        self.y = y
        self.functional_cols = functional_cols or []
        self.continuous_cols = continuous_cols or []
        self.categorical_cols = categorical_cols or []
        self.dt = dt
        
        # 编码分类数据
        self.encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col])
            self.encoders[col] = le
            
        # 设置默认权重
        if self.weights is None:
            n_func = len(self.functional_cols)
            n_cont = len(self.continuous_cols) 
            n_cat = len(self.categorical_cols)
            total = n_func + n_cont + n_cat
            self.weights = {
                'functional': 1.0/n_func if n_func > 0 else 0,
                'continuous': 1.0/n_cont if n_cont > 0 else 0,
                'categorical': 1.0/n_cat if n_cat > 0 else 0
            }
            
        return self
    
    def predict(self, X_test):
        """预测新数据"""
        X_test = X_test.copy()
        
        # 编码测试数据的分类变量
        for col in self.categorical_cols:
            if col in self.encoders:
                # 处理未见过的类别
                mask = ~X_test[col].isin(self.encoders[col].classes_)
                if mask.any():
                    X_test.loc[mask, col] = self.encoders[col].classes_[0]
                X_test[col] = self.encoders[col].transform(X_test[col])
        
        predictions = []
        for _, test_row in X_test.iterrows():
            distances = []
            
            # 计算与所有训练样本的距离
            for idx, train_row in self.X.iterrows():
                dist = self._mixed_distance(train_row, test_row)
                distances.append((dist, self.y.iloc[idx]))
            
            # 找到k个最近邻
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.k]
            
            # 回归：取平均值，分类：取众数
            if self.y.dtype == 'object' or len(np.unique(self.y)) < 10:
                # 分类问题
                neighbor_classes = [neighbor[1] for neighbor in neighbors]
                prediction = max(set(neighbor_classes), key=neighbor_classes.count)
            else:
                # 回归问题
                neighbor_values = [neighbor[1] for neighbor in neighbors]
                prediction = np.mean(neighbor_values)
                
            predictions.append(prediction)
            
        return np.array(predictions)

# ==================== 示例使用代码 ====================

def generate_sample_data(n_samples=100):
    """生成示例混合数据类型"""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        # 函数型数据：时间序列
        t = np.linspace(0, 1, 50)
        functional_data1 = np.sin(2 * np.pi * t) + 0.1 * np.random.normal(size=50)
        functional_data2 = np.cos(2 * np.pi * t) + 0.1 * np.random.normal(size=50)
        
        # 连续数据
        continuous1 = np.random.normal(0, 1)
        continuous2 = np.random.normal(5, 2)
        
        # 分类数据
        categorical1 = np.random.choice(['A', 'B', 'C'])
        categorical2 = np.random.choice(['X', 'Y'])
        
        # 目标变量（回归或分类）
        target_reg = (continuous1 * 2 + continuous2 * 0.5 + 
                     np.mean(functional_data1) + np.random.normal(0, 0.1))
        target_clf = 'Class_' + str(i % 3)
        
        data.append({
            'func1': functional_data1,
            'func2': functional_data2,
            'cont1': continuous1,
            'cont2': continuous2, 
            'cat1': categorical1,
            'cat2': categorical2,
            'target_reg': target_reg,
            'target_clf': target_clf
        })
    
    return pd.DataFrame(data)

# 生成示例数据
df = generate_sample_data(200)

# 准备特征和目标变量
X = df[['func1', 'func2', 'cont1', 'cont2', 'cat1', 'cat2']]
y_reg = df['target_reg']  # 用于回归
y_clf = df['target_clf']  # 用于分类

# 初始化模型
model_reg = FunctionalKNN(k=5, weights={'functional': 0.4, 'continuous': 0.4, 'categorical': 0.2})

# 拟合回归模型
model_reg.fit(X, y_reg, 
              functional_cols=['func1', 'func2'],
              continuous_cols=['cont1', 'cont2'], 
              categorical_cols=['cat1', 'cat2'],
              dt=0.02)

# 预测（使用部分数据作为测试）
X_test = X.iloc[:10]
predictions_reg = model_reg.predict(X_test)

print("回归预测结果:")
print(predictions_reg)

# 分类模型
model_clf = FunctionalKNN(k=5)
model_clf.fit(X, y_clf,
              functional_cols=['func1', 'func2'],
              continuous_cols=['cont1', 'cont2'],
              categorical_cols=['cat1', 'cat2'],
              dt=0.02)

predictions_clf = model_clf.predict(X_test)
print("\n分类预测结果:")
print(predictions_clf)