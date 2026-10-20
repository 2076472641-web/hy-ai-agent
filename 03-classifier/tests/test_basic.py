# -*- coding: utf-8 -*-
"""
基础冒烟测试 (pytest)
运行方式：pytest tests/test_basic.py -v
覆盖内容：
    1. 模拟数据生成（每类≥10条，快速运行）
    2. 数据清洗流程
    3. 中文分词器
    4. TF-IDF特征提取
    5. RandomForestClassifier（小数据快速训练）
    6. FastTextClassifier（小数据快速训练）
    7. Evaluator计算F1等指标
"""

import sys
import os
from pathlib import Path

import pytest
import numpy as np
import pandas as pd


# ============================================================================
# 全局配置 & Fixtures
# ============================================================================

SAMPLES_PER_CATEGORY = 15  # 每类样本数（≥10条，用于快速测试）
SEED = 42
NUM_CATEGORIES = 12


@pytest.fixture(scope="session")
def project_setup():
    """会话级fixture：初始化日志、种子、目录"""
    from src.utils import setup_logger, seed_everything, ensure_project_dirs
    from loguru import logger

    logger.remove()
    setup_logger(log_level="WARNING", console_output=True, file_output=False)
    seed_everything(SEED)
    ensure_project_dirs()
    return True


@pytest.fixture(scope="session")
def mock_raw_data(project_setup):
    """生成模拟原始数据（会话级复用）"""
    from src.data.data_loader import MockDataGenerator, load_categories_config

    categories = load_categories_config()
    generator = MockDataGenerator(
        categories=categories,
        random_seed=SEED,
        samples_per_category=SAMPLES_PER_CATEGORY,
    )
    df = generator.generate()
    return df


@pytest.fixture(scope="session")
def clean_data(mock_raw_data):
    """生成预处理后的干净数据"""
    from src.data.preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor()
    df_clean = preprocessor.transform(mock_raw_data)
    return df_clean


# ============================================================================
# Test 1: 模拟数据生成
# ============================================================================

class TestMockDataGenerator:
    """测试模拟数据生成"""

    def test_data_shape(self, mock_raw_data):
        """测试数据量：总样本数 = 类目数 × 每类样本数"""
        total_expected = NUM_CATEGORIES * SAMPLES_PER_CATEGORY
        assert len(mock_raw_data) == total_expected, (
            f"期望 {total_expected} 条，实际 {len(mock_raw_data)} 条"
        )

    def test_required_columns(self, mock_raw_data):
        """测试必需列存在"""
        required = {"title", "detail", "category_id", "category_name"}
        missing = required - set(mock_raw_data.columns)
        assert not missing, f"缺少必需列: {missing}"

    def test_category_distribution(self, mock_raw_data):
        """测试每个类目样本数≥10且分布均匀"""
        counts = mock_raw_data["category_id"].value_counts().sort_index()
        assert len(counts) == NUM_CATEGORIES, f"类目数应为 {NUM_CATEGORIES}，实际 {len(counts)}"
        for cat_id, count in counts.items():
            assert count >= 10, f"类目{cat_id}样本数={count}，小于10条"
            # 均匀性：所有类目数量相同
            assert count == SAMPLES_PER_CATEGORY, (
                f"类目{cat_id}样本数={count}，期望 {SAMPLES_PER_CATEGORY}"
            )

    def test_text_non_empty(self, mock_raw_data):
        """测试标题和详情非空"""
        empty_title = mock_raw_data["title"].fillna("").str.strip().eq("").sum()
        empty_detail = mock_raw_data["detail"].fillna("").str.strip().eq("").sum()
        assert empty_title == 0, f"有 {empty_title} 条空标题"
        assert empty_detail == 0, f"有 {empty_detail} 条空详情"

    def test_category_mapping_consistency(self, mock_raw_data):
        """测试category_id和category_name对应关系"""
        pairs = mock_raw_data[["category_id", "category_name"]].drop_duplicates()
        # 每个ID仅对应一个名称
        id_counts = pairs["category_id"].value_counts()
        assert (id_counts == 1).all(), "存在同一ID对应多个名称的情况"
        # 每个名称仅对应一个ID
        name_counts = pairs["category_name"].value_counts()
        assert (name_counts == 1).all(), "存在同一名称对应多个ID的情况"


# ============================================================================
# Test 2: 数据清洗流程
# ============================================================================

class TestDataPreprocessor:
    """测试数据清洗流程"""

    def test_no_duplicates_after_clean(self, clean_data, mock_raw_data):
        """测试清洗后无重复（基于title+detail）"""
        before = len(mock_raw_data)
        after = len(clean_data)
        # 模拟数据可能没有重复，数量应相等或略少
        assert after <= before, f"清洗后数据({after})居然比原始({before})多！"
        dup_keys = clean_data.duplicated(subset=["title", "detail"], keep=False)
        assert dup_keys.sum() == 0, "清洗后仍有重复数据"

    def test_no_missing_text(self, clean_data):
        """测试清洗后无空文本"""
        for col in ["title", "detail"]:
            empty = clean_data[col].fillna("").str.strip().eq("").sum()
            assert empty == 0, f"列 {col} 有 {empty} 条空值"

    def test_cleaned_columns(self, clean_data):
        """测试预处理后列齐全"""
        expected = {
            "title", "detail", "category_id", "category_name",
            "title_clean", "detail_clean",
            "title_tokens", "detail_tokens",
            "combined_text",
        }
        missing = expected - set(clean_data.columns)
        assert not missing, f"预处理后缺少列: {missing}"

    def test_combined_text_not_empty(self, clean_data):
        """测试combined_text非空"""
        empty = clean_data["combined_text"].fillna("").str.strip().eq("").sum()
        assert empty == 0, f"combined_text 有 {empty} 条空值"

    def test_html_removal(self, project_setup):
        """测试HTML标签移除"""
        from src.data.preprocessor import TextCleaner
        cleaner = TextCleaner()
        raw = "<p>这是<b>加粗</b>文本<br/>链接: <a href='x'>点我</a></p>"
        cleaned = cleaner.clean_text(raw)
        assert "<" not in cleaned, "HTML标签未移除"
        assert "加粗" in cleaned and "文本" in cleaned

    def test_emoji_removal(self, project_setup):
        """测试emoji移除"""
        from src.data.preprocessor import TextCleaner
        cleaner = TextCleaner()
        raw = "好产品👍 推荐⭐⭐⭐ 包邮🚚"
        cleaned = cleaner.clean_text(raw)
        # emoji应被移除
        has_emoji = any(ord(c) > 0x1F000 for c in cleaned)
        assert not has_emoji, f"仍包含emoji: {cleaned}"


# ============================================================================
# Test 3: 中文分词器
# ============================================================================

class TestChineseTokenizer:
    """测试中文分词器"""

    def test_tokenizer_basic(self, project_setup):
        """测试基本分词效果"""
        from src.data.preprocessor import ChineseTokenizer
        tok = ChineseTokenizer()
        text = "夏季新款连衣裙女装纯棉短袖"
        tokens = tok.tokenize(text)
        assert isinstance(tokens, list), "分词结果应为列表"
        assert len(tokens) > 0, "分词结果为空"
        # 常见词应被正确切分
        joined = " ".join(tokens)
        assert "连衣裙" in joined or "新款" in joined, f"分词效果不佳: {tokens}"

    def test_tokenizer_no_empty_tokens(self, project_setup):
        """测试无空token和纯空白token"""
        from src.data.preprocessor import ChineseTokenizer
        tok = ChineseTokenizer()
        tokens = tok.tokenize("  你好， 世界！  ")
        for t in tokens:
            assert t.strip() != "", f"存在空token: {repr(tokens)}"

    def test_stopwords_removal(self, project_setup):
        """测试停用词移除"""
        from src.data.preprocessor import ChineseTokenizer
        tok = ChineseTokenizer()
        # "的"和"是"都是停用词
        text = "这款的产品是非常好用的"
        tokens = tok.tokenize(text)
        assert "的" not in tokens, "停用词'的'未移除"
        assert "是" not in tokens, "停用词'是'未移除"
        assert len(tokens) > 0, "移除停用词后不应为空"

    def test_tokenizer_on_sample_data(self, clean_data):
        """在真实样例上测试分词结果"""
        tokens_series = clean_data["title_tokens"].dropna()
        # 至少有一些分词结果
        avg_len = tokens_series.apply(lambda x: len(x) if isinstance(x, list) else len(str(x).split())).mean()
        assert avg_len > 1, f"分词平均长度过短: {avg_len}"

    def test_tokenize_method_list_input(self, project_setup):
        """测试批量分词"""
        from src.data.preprocessor import ChineseTokenizer
        tok = ChineseTokenizer()
        texts = ["夏季新款连衣裙", "手机数码产品", "美妆护肤精华液"]
        result = tok.tokenize_batch(texts)
        assert isinstance(result, list), "批量分词应返回列表"
        assert len(result) == 3, f"期望3条结果，实际{len(result)}"
        for r in result:
            assert isinstance(r, list), "每条结果应为token列表"


# ============================================================================
# Test 4: TF-IDF特征提取
# ============================================================================

class TestTFIDFVectorizer:
    """测试TF-IDF特征提取"""

    def test_tfidf_fit_transform(self, clean_data, project_setup):
        """测试TF-IDF拟合和转换"""
        from src.features.text_vectorizer import TextVectorizer
        vec = TextVectorizer(max_features=500, ngram_range=(1, 1))
        texts = clean_data["combined_text"].fillna("").tolist()
        X = vec.fit_transform(texts)
        assert X is not None, "TF-IDF转换结果为None"
        # 稀疏矩阵shape应为 (样本数, 特征数)
        assert X.shape[0] == len(texts), f"样本数不匹配: {X.shape[0]} vs {len(texts)}"
        assert X.shape[1] <= 500, f"特征数超过max_features: {X.shape[1]}"

    def test_tfidf_vocabulary(self, clean_data, project_setup):
        """测试词表非空且有常见商品词汇"""
        from src.features.text_vectorizer import TextVectorizer
        vec = TextVectorizer(max_features=1000)
        texts = clean_data["combined_text"].fillna("").tolist()
        vec.fit(texts)
        vocab = vec.get_vocabulary()
        assert isinstance(vocab, dict), "词表应为字典"
        assert len(vocab) > 50, f"词表太小: {len(vocab)}"
        # 词表ID应为整数
        for word, idx in vocab.items():
            assert isinstance(idx, int), f"词表ID非整数: {word}->{idx}"

    def test_tfidf_save_load(self, clean_data, tmp_path, project_setup):
        """测试TF-IDF模型保存和加载"""
        from src.features.text_vectorizer import TextVectorizer
        import numpy as np

        vec1 = TextVectorizer(max_features=300)
        texts = clean_data["combined_text"].fillna("").tolist()[:30]
        X1 = vec1.fit_transform(texts)

        save_path = tmp_path / "tfidf_test.pkl"
        vec1.save(str(save_path))
        assert save_path.exists(), "TF-IDF保存失败"

        vec2 = TextVectorizer()
        vec2.load(str(save_path))
        X2 = vec2.transform(texts)

        # 转换结果应相同
        assert X1.shape == X2.shape, "加载后shape不一致"
        if hasattr(X1, "toarray"):
            np.testing.assert_array_almost_equal(
                X1.toarray(), X2.toarray(),
                err_msg="加载后转换结果不一致"
            )

    def test_tfidf_feature_names(self, clean_data, project_setup):
        """测试特征名获取"""
        from src.features.text_vectorizer import TextVectorizer
        vec = TextVectorizer(max_features=100)
        texts = clean_data["combined_text"].fillna("").tolist()
        vec.fit(texts)
        feature_names = vec.get_feature_names()
        assert isinstance(feature_names, list), "特征名应为列表"
        assert len(feature_names) > 0, "特征名列表为空"
        assert len(feature_names) <= 100, "特征数超过max_features"


# ============================================================================
# Test 5: RandomForestClassifier
# ============================================================================

class TestRandomForestClassifier:
    """测试RandomForest分类器"""

    @pytest.fixture
    def rf_train_data(self, clean_data):
        """准备RF训练数据（小样本）"""
        X = clean_data["combined_text"].fillna("").tolist()
        y = clean_data["category_id"].astype(int).tolist()
        return X, y

    def test_rf_fit_predict(self, rf_train_data, project_setup):
        """测试RF训练和预测基本流程"""
        from src.models.random_forest import RandomForestClassifier
        X, y = rf_train_data
        model = RandomForestClassifier(
            n_estimators=30,
            max_depth=15,
            random_state=SEED,
            tfidf_max_features=500,
        )
        model.fit(X, y)

        # 预测
        preds = model.predict(X[:20])
        assert isinstance(preds, (list, np.ndarray)), "预测结果类型错误"
        assert len(preds) == 20, f"预测数量不匹配: {len(preds)}"
        # 预测值应为有效类别ID
        for p in preds:
            assert 0 <= int(p) < NUM_CATEGORIES, f"无效预测类别: {p}"

    def test_rf_predict_proba(self, rf_train_data, project_setup):
        """测试RF预测概率"""
        from src.models.random_forest import RandomForestClassifier
        X, y = rf_train_data
        model = RandomForestClassifier(
            n_estimators=30, max_depth=15, random_state=SEED,
            tfidf_max_features=300,
        )
        model.fit(X, y)
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, NUM_CATEGORIES), f"概率shape错误: {proba.shape}"
        # 每行概率和应为1
        row_sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(
            row_sums, np.ones(10), decimal=2,
            err_msg="概率行和不为1"
        )

    def test_rf_save_load(self, rf_train_data, tmp_path, project_setup):
        """测试RF模型保存加载"""
        from src.models.random_forest import RandomForestClassifier
        X, y = rf_train_data

        m1 = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=SEED, tfidf_max_features=300)
        m1.fit(X, y)
        p1 = m1.predict(X[:10])

        save_dir = tmp_path / "rf_model"
        save_dir.mkdir()
        m1.save(str(save_dir))
        assert (save_dir / "model.pkl").exists(), "模型文件未保存"
        assert (save_dir / "tfidf.pkl").exists(), "TF-IDF未保存"

        m2 = RandomForestClassifier()
        m2.load(str(save_dir))
        p2 = m2.predict(X[:10])

        np.testing.assert_array_equal(p1, p2, err_msg="加载后预测不一致")

    def test_rf_minimum_accuracy(self, rf_train_data, project_setup):
        """测试RF训练后在训练集上达到最低准确率（≥20%，优于随机）"""
        from src.models.random_forest import RandomForestClassifier
        X, y = rf_train_data
        model = RandomForestClassifier(
            n_estimators=50, max_depth=20, random_state=SEED,
            tfidf_max_features=800,
        )
        model.fit(X, y)
        preds = model.predict(X)
        acc = np.mean(np.array(preds) == np.array(y))
        random_baseline = 1.0 / NUM_CATEGORIES  # 约8.3%
        assert acc > random_baseline * 2, (
            f"训练准确率 {acc:.3f} 太低（随机={random_baseline:.3f}×2={random_baseline*2:.3f}）"
        )


# ============================================================================
# Test 6: FastTextClassifier
# ============================================================================

class TestFastTextClassifier:
    """测试FastText分类器"""

    @pytest.fixture
    def ft_train_data(self, clean_data):
        """准备FT训练数据"""
        X = clean_data["combined_text"].fillna("").tolist()
        y = clean_data["category_id"].astype(int).tolist()
        return X, y

    def test_ft_fit_predict(self, ft_train_data, project_setup):
        """测试FT训练和预测"""
        from src.models.fasttext_model import FastTextClassifier
        X, y = ft_train_data
        model = FastTextClassifier(
            epoch=10,
            lr=0.1,
            dim=50,
            wordNgrams=1,
            random_state=SEED,
        )
        model.fit(X, y)

        preds = model.predict(X[:20])
        assert isinstance(preds, (list, np.ndarray)), "预测结果类型错误"
        assert len(preds) == 20, f"预测数量不匹配: {len(preds)}"
        for p in preds:
            assert 0 <= int(p) < NUM_CATEGORIES, f"无效预测类别: {p}"

    def test_ft_predict_proba(self, ft_train_data, project_setup):
        """测试FT预测概率"""
        from src.models.fasttext_model import FastTextClassifier
        X, y = ft_train_data
        model = FastTextClassifier(epoch=10, lr=0.1, dim=30, random_state=SEED)
        model.fit(X, y)
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, NUM_CATEGORIES), f"概率shape错误: {proba.shape}"
        row_sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(
            row_sums, np.ones(10), decimal=2,
            err_msg="概率行和不为1"
        )

    def test_ft_save_load(self, ft_train_data, tmp_path, project_setup):
        """测试FT模型保存加载"""
        from src.models.fasttext_model import FastTextClassifier
        X, y = ft_train_data

        m1 = FastTextClassifier(epoch=10, lr=0.1, dim=30, random_state=SEED)
        m1.fit(X, y)
        p1 = m1.predict(X[:10])

        save_dir = tmp_path / "ft_model"
        save_dir.mkdir()
        m1.save(str(save_dir))
        assert (save_dir / "model.bin").exists(), "FT模型bin文件未保存"

        m2 = FastTextClassifier()
        m2.load(str(save_dir))
        p2 = m2.predict(X[:10])

        np.testing.assert_array_equal(p1, p2, err_msg="FT加载后预测不一致")

    def test_ft_minimum_accuracy(self, ft_train_data, project_setup):
        """测试FT训练后在训练集达到最低准确率"""
        from src.models.fasttext_model import FastTextClassifier
        X, y = ft_train_data
        model = FastTextClassifier(
            epoch=20, lr=0.2, dim=50, wordNgrams=2,
            random_state=SEED,
        )
        model.fit(X, y)
        preds = model.predict(X)
        acc = np.mean(np.array(preds) == np.array(y))
        random_baseline = 1.0 / NUM_CATEGORIES
        assert acc > random_baseline * 2, (
            f"FT训练准确率 {acc:.3f} 太低（随机×2={random_baseline*2:.3f}）"
        )


# ============================================================================
# Test 7: Evaluator计算F1等指标
# ============================================================================

class TestEvaluator:
    """测试评估器计算F1等指标"""

    @pytest.fixture
    def trained_rf(self, clean_data):
        """训练好的RF模型（用于评估测试）"""
        from src.models.random_forest import RandomForestClassifier
        X = clean_data["combined_text"].fillna("").tolist()
        y = clean_data["category_id"].astype(int).tolist()
        model = RandomForestClassifier(
            n_estimators=50, max_depth=20, random_state=SEED,
            tfidf_max_features=500,
        )
        model.fit(X, y)
        return model, X, y

    def test_evaluator_basic_metrics(self, trained_rf, tmp_path, project_setup):
        """测试评估器计算Accuracy/Precision/Recall/F1等"""
        from src.evaluation.evaluator import Evaluator
        from src.utils import get_category_mappings

        model, X, y = trained_rf
        _, _, class_names = get_category_mappings()

        evaluator = Evaluator(
            class_names=class_names,
            output_dir=str(tmp_path),
            compute_roc=False,
            compute_pr=False,
        )
        result = evaluator.evaluate(model, X, y, model_name="test_rf")

        # 基本属性存在
        assert result.model_name == "test_rf"
        assert len(result.y_true) == len(y)
        assert len(result.y_pred) == len(y)
        assert result.num_samples == len(y)
        assert result.num_classes == NUM_CATEGORIES

        # 指标合理范围 [0, 1]
        for metric_name in ["accuracy", "precision_macro", "recall_macro",
                            "f1_macro", "f1_micro"]:
            val = getattr(result, metric_name)
            assert 0.0 <= val <= 1.0, f"{metric_name}={val} 超出 [0,1] 范围"

        # Accuracy 应合理（至少好于随机）
        random_acc = 1.0 / NUM_CATEGORIES
        assert result.accuracy > random_acc, (
            f"Accuracy={result.accuracy:.3f} 低于随机基线 {random_acc:.3f}"
        )

    def test_evaluator_per_class_metrics(self, trained_rf, tmp_path, project_setup):
        """测试每个类别的指标"""
        from src.evaluation.evaluator import Evaluator
        from src.utils import get_category_mappings

        model, X, y = trained_rf
        _, _, class_names = get_category_mappings()

        evaluator = Evaluator(
            class_names=class_names, output_dir=str(tmp_path),
            compute_roc=False, compute_pr=False,
        )
        result = evaluator.evaluate(model, X, y)

        df_per_class = result.get_per_class_metrics_df()
        assert isinstance(df_per_class, pd.DataFrame), "per-class结果应为DataFrame"
        assert len(df_per_class) == NUM_CATEGORIES, f"应为{NUM_CATEGORIES}行，实际{len(df_per_class)}"

        required_cols = {"precision", "recall", "f1", "support"}
        missing = required_cols - set(df_per_class.columns)
        assert not missing, f"per-class缺少列: {missing}"

        # support总和应为样本数
        total_support = df_per_class["support"].sum()
        assert total_support == len(y), f"support总和={total_support} != 样本数{len(y)}"

    def test_evaluator_confusion_matrix(self, trained_rf, tmp_path, project_setup):
        """测试混淆矩阵生成"""
        from src.evaluation.evaluator import Evaluator
        from src.utils import get_category_mappings

        model, X, y = trained_rf
        _, _, class_names = get_category_mappings()

        evaluator = Evaluator(
            class_names=class_names, output_dir=str(tmp_path),
            compute_roc=False, compute_pr=False,
        )
        result = evaluator.evaluate(model, X, y)

        cm = result.get_confusion_matrix()
        assert cm.shape == (NUM_CATEGORIES, NUM_CATEGORIES), (
            f"混淆矩阵shape错误: {cm.shape}"
        )
        # 混淆矩阵总和 = 样本数
        assert cm.sum() == len(y), f"混淆矩阵总和={cm.sum()} != 样本数{len(y)}"
        # 对角线元素之和 ≤ 总样本数（不可能超过）
        assert np.trace(cm) <= len(y), "混淆矩阵对角线和超过样本数"

    def test_evaluator_save_report(self, trained_rf, tmp_path, project_setup):
        """测试评估报告保存"""
        from src.evaluation.evaluator import Evaluator
        from src.utils import get_category_mappings

        model, X, y = trained_rf
        _, _, class_names = get_category_mappings()

        evaluator = Evaluator(
            class_names=class_names, output_dir=str(tmp_path),
            compute_roc=False, compute_pr=False,
        )
        result = evaluator.evaluate(model, X, y)

        saved = result.save(str(tmp_path), prefix="test")
        assert isinstance(saved, dict), "save应返回字典"
        for name, path_str in saved.items():
            p = Path(path_str)
            assert p.exists(), f"报告文件{name}未生成: {p}"


# ============================================================================
# Test 8: utils工具函数快速自测
# ============================================================================

class TestUtils:
    """测试utils中的关键工具"""

    def test_category_mappings(self, project_setup):
        """测试类目映射"""
        from src.utils import get_category_mappings, DEFAULT_CATEGORY_NAMES
        id2name, name2id, names = get_category_mappings()

        assert len(names) == NUM_CATEGORIES, f"类目数量应为{NUM_CATEGORIES}"
        assert len(id2name) == NUM_CATEGORIES
        assert len(name2id) == NUM_CATEGORIES

        # 双向映射一致性
        for i in range(NUM_CATEGORIES):
            name = id2name[i]
            assert name2id[name] == i, f"映射不一致: id={i} name={name}"

        # 默认名称包含
        for default_name in DEFAULT_CATEGORY_NAMES:
            assert default_name in name2id, f"缺少默认类目: {default_name}"

    def test_timer_context(self, project_setup):
        """测试Timer上下文管理器"""
        from src.utils import Timer
        import time
        with Timer("test_timer", verbose=False) as t:
            time.sleep(0.05)
        assert 0.04 <= t["elapsed"] <= 0.5, f"Timer计时异常: {t['elapsed']}"

    def test_seed_everything(self, project_setup):
        """测试种子固定"""
        from src.utils import seed_everything
        seed_everything(123)
        import random
        r1 = random.random()
        n1 = np.random.rand()
        seed_everything(123)
        r2 = random.random()
        n2 = np.random.rand()
        assert r1 == r2, "random种子未固定"
        assert n1 == n2, "numpy种子未固定"

    def test_lazy_property(self, project_setup):
        """测试惰性属性装饰器"""
        from src.utils import lazy_property

        call_count = {"n": 0}

        class Foo:
            @lazy_property
            def expensive(self):
                call_count["n"] += 1
                return 42

        obj = Foo()
        _ = obj.expensive
        _ = obj.expensive
        _ = obj.expensive
        assert call_count["n"] == 1, "lazy_property计算了多次"
        assert obj.expensive == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
