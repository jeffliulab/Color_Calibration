"""
XGBoost参数贝叶斯优化 - CUDA加速版
使用贝叶斯优化高效搜索XGBoost的最佳参数
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from datetime import timedelta
from tqdm import tqdm
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skopt.callbacks import VerboseCallback
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from skimage import color
from src.train.pre_train import load_preprocessed_data
import os
os.environ['XGBOOST_USE_CUDA'] = '1'
# 首先导入warnings模块并过滤XGBoost警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# ================================
# 1. 定义路径和配置
# ================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_CONFIG = {
    "model_type": "xgboost",
    "version": "v2_bayesian_cuda",
    "extension": "pkl",
}

MODEL_PATH = ROOT_DIR / "data/models" / MODEL_CONFIG["model_type"] / f"model_{MODEL_CONFIG['version']}.{MODEL_CONFIG['extension']}"
RESULTS_PATH = ROOT_DIR / "data/models" / MODEL_CONFIG["model_type"] / f"optimization_results_{MODEL_CONFIG['version']}.csv"

# ================================
# 2. GPU检测函数
# ================================
def check_gpu_availability():
    """
    检查CUDA GPU是否可用于XGBoost
    
    Returns:
        bool: GPU是否可用
    """
    print("检查CUDA GPU是否可用...")
    try:
        import xgboost as xgb
        
        # 创建一个小数据集进行测试
        X_test = np.random.random((10, 5))
        y_test = np.random.random(10)
        
        # 尝试使用GPU训练一个小模型
        model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=10)
        model.fit(X_test, y_test)
        
        print("✓ CUDA GPU可用，将使用GPU加速训练")
        return True
    except Exception as e:
        print(f"✗ 无法使用CUDA GPU: {e}")
        print("脚本将终止")
        return False

# ================================
# 3. 贝叶斯参数优化
# ================================
def bayesian_optimize_xgboost(n_iter=50, cv=3, n_points=1):
    """
    使用贝叶斯优化寻找XGBoost的最佳参数

    Args:
        n_iter (int): 迭代次数
        cv (int): 交叉验证折数
        n_points (int): 每次迭代评估的点数

    Returns:
        tuple: (best_params, results_df, best_estimator)
    """
    print("=" * 50)
    print("贝叶斯参数优化")
    print("=" * 50)
    
    # 记录开始时间
    start_time = time.time()
    
    # 加载数据
    X, y = load_preprocessed_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 设置交叉验证
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 定义搜索空间
    search_spaces = {
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'n_estimators': Integer(100, 1000),
        'max_depth': Integer(3, 10),
        'min_child_weight': Integer(1, 10),
        'gamma': Real(0, 0.5),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'reg_alpha': Real(1e-4, 1, prior='log-uniform'),
        'reg_lambda': Real(1e-4, 10, prior='log-uniform')
    }
    
    # 创建基础模型
    base_model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',
        random_state=42
    )
    
    # 创建贝叶斯搜索对象
    bayes_search = BayesSearchCV(
        estimator=base_model,
        search_spaces=search_spaces,
        n_iter=n_iter,
        n_points=n_points,
        scoring='neg_mean_squared_error',
        cv=kfold,
        verbose=2,  # 较少的详细输出，避免与进度条冲突
        random_state=42,
        n_jobs=1  # 使用GPU时设为1
    )
    
    # 执行搜索
    print(f"开始贝叶斯参数搜索 ({n_iter} 次迭代, {cv} 折交叉验证)...")
    
    # 创建进度条
    progress_bar = tqdm(total=n_iter, desc="贝叶斯优化进度", position=0)
    
    # 创建回调函数用于更新进度条
    class ProgressCallback:
        def __init__(self, pbar):
            self.pbar = pbar
            self.iter_count = 0
            self.start_time = time.time()
            
        def __call__(self, res):
            if res.func_vals is not None and len(res.func_vals) > self.iter_count:
                new_iters = len(res.func_vals) - self.iter_count
                self.pbar.update(new_iters)
                self.iter_count = len(res.func_vals)
                
                # 估计剩余时间
                if self.iter_count > 0:
                    elapsed = time.time() - self.start_time
                    remaining = (elapsed / self.iter_count) * (n_iter - self.iter_count)
                    self.pbar.set_postfix({
                        "剩余时间": f"{timedelta(seconds=int(remaining))}",
                        "已完成": f"{self.iter_count}/{n_iter}"
                    })
    
    # 创建回调
    verbose_callback = VerboseCallback(1)
    progress_callback = ProgressCallback(progress_bar)
    
    try:
        # 执行带回调的搜索
        results = bayes_search.fit(X_train, y_train, callback=[verbose_callback, progress_callback])
    finally:
        progress_bar.close()
    
    # 获取结果
    results_df = pd.DataFrame(bayes_search.cv_results_)
    best_params = bayes_search.best_params_
    
    # 显示总耗时
    elapsed = time.time() - start_time
    print(f"\n参数搜索完成，总耗时: {str(timedelta(seconds=int(elapsed)))}")
    
    # 打印最佳参数
    print("\n最佳参数:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # 保存结果
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"搜索结果已保存至 {RESULTS_PATH}")
    
    # 评估最佳模型
    y_pred = bayes_search.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n测试集性能: RMSE = {rmse:.4f}, R² = {r2:.4f}")
    
    return best_params, results_df, bayes_search.best_estimator_

# ================================
# 4. 训练最终模型
# ================================
def train_final_model(best_params):
    """
    使用最佳参数训练最终模型
    
    Args:
        best_params (dict): 最佳参数
    
    Returns:
        tuple: (model, X_test, y_test, y_pred)
    """
    print("\n" + "=" * 50)
    print("训练最终模型")
    print("=" * 50)
    
    # 记录开始时间
    start_time = time.time()
    
    # 加载数据
    X, y = load_preprocessed_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建并训练模型
    model = XGBRegressor(
        objective='reg:squarederror', 
        tree_method='hist',
        device='cuda',
        random_state=42, 
        **best_params
    )
    
    # 训练模型并显示进度
    print(f"训练XGBoost模型，使用最佳参数...")
    
    # 使用进度条
    train_progress = tqdm(total=best_params['n_estimators'], desc="训练进度")
    
    class ProgressCallback:
        def __init__(self, pbar):
            self.pbar = pbar
            self.prev_trees = 0
            
        def __call__(self, env):
            current_trees = env.iteration
            trees_added = current_trees - self.prev_trees
            self.pbar.update(trees_added)
            self.prev_trees = current_trees
    
    try:
        model.fit(
            X_train, 
            y_train, 
            verbose=False,
            callbacks=[ProgressCallback(train_progress)]
        )
    finally:
        train_progress.close()
    
    # 预测测试集
    y_pred = model.predict(X_test)
    
    # 保存模型
    joblib.dump(model, MODEL_PATH)
    print(f"最终模型已保存至 {MODEL_PATH}")
    
    # 保存最终参数
    pd.DataFrame([best_params]).to_csv(
        ROOT_DIR / "data/models" / MODEL_CONFIG["model_type"] / f"final_params_{MODEL_CONFIG['version']}.csv", 
        index=False
    )
    
    # 显示总耗时
    elapsed = time.time() - start_time
    print(f"最终模型训练完成，耗时: {str(timedelta(seconds=int(elapsed)))}")
    
    return model, X_test, y_test, y_pred

# ================================
# 5. 评估模型并分析特征重要性
# ================================
def evaluate_and_analyze(model, X_test, y_test, y_pred, X):
    """
    评估模型性能并分析特征重要性
    
    Args:
        model: 训练好的XGBoost模型
        X_test (DataFrame): 测试特征
        y_test (DataFrame): 测试标签
        y_pred (ndarray): 预测结果
        X (DataFrame): 全部特征数据
    """
    print("\n" + "=" * 50)
    print("模型评估")
    print("=" * 50)
    
    # 计算性能指标
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # 计算Lab颜色空间差异
    y_test_norm = y_test.to_numpy() / 255.0
    y_pred_norm = y_pred / 255.0
    
    y_test_lab = color.rgb2lab(y_test_norm.reshape(-1, 1, 3)).reshape(-1, 3)
    y_pred_lab = color.rgb2lab(y_pred_norm.reshape(-1, 1, 3)).reshape(-1, 3)
    
    delta_e_lab = np.sqrt(np.sum((y_test_lab - y_pred_lab) ** 2, axis=1))
    mean_delta_e_lab = np.mean(delta_e_lab)
    median_delta_e_lab = np.median(delta_e_lab)
    
    print(f"Mean ΔE (Lab): {mean_delta_e_lab:.2f}")
    print(f"Median ΔE (Lab): {median_delta_e_lab:.2f}")
    
    # 特征重要性分析
    print("\n特征重要性分析:")
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # 显示前10个重要特征
    print("\n前10个重要特征:")
    print(feature_importance.head(10))
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "data/models" / MODEL_CONFIG["model_type"] / f"feature_importance_{MODEL_CONFIG['version']}.png")
    plt.show()
    
    # 保存评估结果
    evaluation = {
        'r2': r2,
        'rmse': rmse,
        'mean_delta_e_lab': mean_delta_e_lab,
        'median_delta_e_lab': median_delta_e_lab
    }
    
    pd.DataFrame([evaluation]).to_csv(
        ROOT_DIR / "data/models" / MODEL_CONFIG["model_type"] / f"evaluation_{MODEL_CONFIG['version']}.csv",
        index=False
    )
    
    return evaluation

# ================================
# 6. 主函数
# ================================
def main():
    """
    主函数，使用贝叶斯优化寻找XGBoost最佳参数
    """
    # 记录总开始时间
    total_start_time = time.time()
    
    # 检查GPU是否可用
    if not check_gpu_availability():
        return
    
    # 1. 执行贝叶斯参数优化
    best_params, _, _ = bayesian_optimize_xgboost(n_iter=50, cv=3, n_points=1)
    
    # 2. 使用最佳参数训练最终模型
    model, X_test, y_test, y_pred = train_final_model(best_params)
    
    # 3. 评估模型并分析特征重要性
    X, _ = load_preprocessed_data()  # 获取特征名
    evaluate_and_analyze(model, X_test, y_test, y_pred, X)
    
    # 显示总耗时
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 50)
    print(f"整个优化过程完成，总耗时: {str(timedelta(seconds=int(total_elapsed)))}")
    print("=" * 50)
    
    return model

if __name__ == "__main__":
    main()