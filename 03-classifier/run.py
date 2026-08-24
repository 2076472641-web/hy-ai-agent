# -*- coding: utf-8 -*-
"""
项目CLI入口脚本（基于click）
优店汇·商品类目智能研判分类系统 - 命令行工具

使用示例：
    python run.py generate-data --samples 500 --output data/raw/mock_data.csv
    python run.py preprocess --input data/raw/mock_data.csv --output data/processed/clean_data.csv --run-eda
    python run.py train --model random_forest --data data/processed/clean_data.csv --output data/models/random_forest
    python run.py compress --model-path data/models/bert --output data/models/bert_distilled --method distill
    python run.py evaluate --model data/models/bert --data data/processed/clean_data --output logs/evaluation
    python run.py serve --api --port 8000
    python run.py serve --frontend --port 8501
    python run.py pipeline
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click
from loguru import logger
import pandas as pd
import numpy as np

# 初始化基础工具
from src.utils import (
    load_config,
    setup_logger,
    seed_everything,
    get_category_mappings,
    Timer,
    ensure_project_dirs,
)
from src import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR

# 确保目录存在
ensure_project_dirs()
setup_logger()
seed_everything(42)


# ============================================================================
# CLI主组
# ============================================================================

@click.group(
    help="优店汇·商品类目智能研判分类系统 - 命令行工具",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(version="1.0.0", prog_name="商品分类系统")
def cli():
    """商品类目智能分类系统 CLI 入口"""
    pass


# ============================================================================
# 1. generate-data 子命令
# ============================================================================

@cli.command(
    "generate-data",
    help="生成模拟商品数据集（十二大类，每类N条）",
)
@click.option(
    "--samples",
    "samples_per_category",
    type=int,
    default=500,
    show_default=True,
    help="每个类目生成的样本数量（至少10条）",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=lambda: RAW_DATA_DIR / "mock_data.csv",
    show_default="data/raw/mock_data.csv",
    help="输出CSV文件路径",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="随机种子（保证可复现）",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="覆盖已存在的输出文件",
)
def generate_data(samples_per_category: int, output_path: Path, seed: int, force: bool):
    """
    为十二大类商品生成包含标题和详情的模拟数据。

    \b
    示例：
        python run.py generate-data --samples 100
        python run.py generate-data --samples 500 --output data/raw/my_data.csv
    """
    if samples_per_category < 10:
        raise click.BadParameter("每个类目样本数至少为10条")

    output_path = Path(output_path)
    if output_path.exists() and not force:
        click.confirm(
            f"输出文件 {output_path} 已存在，是否覆盖？",
            abort=True,
        )

    click.echo(f"\n{'='*60}")
    click.echo(f"📊 开始生成模拟数据")
    click.echo(f"   - 每个类目样本数: {samples_per_category}")
    click.echo(f"   - 总类目数: 12")
    click.echo(f"   - 预计总样本数: {samples_per_category * 12}")
    click.echo(f"   - 随机种子: {seed}")
    click.echo(f"   - 输出路径: {output_path}")
    click.echo(f"{'='*60}\n")

    with Timer("模拟数据生成", verbose=False) as t:
        from src.data.data_loader import MockDataGenerator, load_categories_config

        categories = load_categories_config()
        generator = MockDataGenerator(
            categories=categories,
            random_seed=seed,
            samples_per_category=samples_per_category,
        )
        df = generator.generate()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    click.echo(f"\n✅ 模拟数据生成完成！")
    click.echo(f"   - 总样本数: {len(df)}")
    click.echo(f"   - 列: {list(df.columns)}")
    click.echo(f"   - 文件大小: {output_path.stat().st_size / 1024:.1f} KB")
    click.echo(f"   - 保存路径: {output_path}")
    click.echo(f"   - 耗时: {t['elapsed']:.2f} 秒")

    # 显示类目分布
    click.echo("\n📋 类目分布：")
    cat_counts = df["category_name"].value_counts().sort_index()
    for cat_name, count in cat_counts.items():
        bar = "█" * min(count // (samples_per_category // 20), 40)
        click.echo(f"   {cat_name:<10s} | {bar} {count}条")


# ============================================================================
# 2. preprocess 子命令
# ============================================================================

@cli.command(
    "preprocess",
    help="数据预处理：清洗+去重+分词+可选EDA",
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="输入CSV文件路径（需包含title和detail列）",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=lambda: PROCESSED_DATA_DIR / "clean_data.csv",
    show_default="data/processed/clean_data.csv",
    help="输出预处理后的CSV文件路径",
)
@click.option(
    "--run-eda",
    is_flag=True,
    default=False,
    help="是否运行探索性数据分析（生成EDA报告图表）",
)
@click.option(
    "--no-dedup",
    is_flag=True,
    default=False,
    help="跳过重复数据去除",
)
@click.option(
    "--no-dropna",
    is_flag=True,
    default=False,
    help="跳过空值数据去除",
)
@click.option(
    "--no-tokenize",
    is_flag=True,
    default=False,
    help="跳过中文分词（仅做清洗不做分词）",
)
@click.option(
    "--eda-output",
    "eda_output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=lambda: LOG_DIR / "eda",
    show_default="logs/eda",
    help="EDA报告输出目录",
)
def preprocess(
    input_path: Path,
    output_path: Path,
    run_eda: bool,
    no_dedup: bool,
    no_dropna: bool,
    no_tokenize: bool,
    eda_output_dir: Path,
):
    """
    对原始商品数据执行完整预处理流水线：去重→去空→文本清洗→分词→合并文本。

    \b
    示例：
        python run.py preprocess --input data/raw/mock_data.csv
        python run.py preprocess --input data/raw/mock_data.csv --run-eda
        python run.py preprocess --input data/raw/raw.csv --output data/processed/clean.csv
    """
    click.echo(f"\n{'='*60}")
    click.echo(f"🔧 开始数据预处理")
    click.echo(f"   - 输入文件: {input_path}")
    click.echo(f"   - 输出文件: {output_path}")
    click.echo(f"   - 去重: {'否' if no_dedup else '是'}")
    click.echo(f"   - 去空值: {'否' if no_dropna else '是'}")
    click.echo(f"   - 分词: {'否' if no_tokenize else '是'}")
    click.echo(f"   - EDA分析: {'是' if run_eda else '否'}")
    click.echo(f"{'='*60}\n")

    with Timer("数据预处理", verbose=False) as t:
        from src.data.data_loader import DataLoader
        from src.data.preprocessor import DataPreprocessor

        # 加载数据
        click.echo("📥 加载原始数据...")
        loader = DataLoader()
        df = loader.load_csv(input_path)
        click.echo(f"   原始数据量: {len(df)} 条")

        # 预处理
        click.echo("🔄 执行预处理流水线...")
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.transform(
            df,
            do_dedup=not no_dedup,
            do_dropna=not no_dropna,
            do_clean=True,
            do_tokenize=not no_tokenize,
            do_combine=True,
        )

        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

        # EDA
        if run_eda:
            click.echo("📊 运行探索性数据分析...")
            try:
                from src.data.eda import run_full_eda
                eda_output_dir = Path(eda_output_dir)
                eda_output_dir.mkdir(parents=True, exist_ok=True)
                run_full_eda(df_clean, output_dir=str(eda_output_dir))
                click.echo(f"   EDA报告已保存到: {eda_output_dir}")
            except Exception as e:
                logger.warning(f"EDA分析失败（可忽略）: {e}")

    click.echo(f"\n✅ 数据预处理完成！")
    click.echo(f"   - 原始: {len(df)} 条 → 处理后: {len(df_clean)} 条")
    click.echo(f"   - 数据保留率: {len(df_clean)/len(df)*100:.1f}%")
    click.echo(f"   - 输出列: {list(df_clean.columns)}")
    click.echo(f"   - 保存路径: {output_path}")
    click.echo(f"   - 耗时: {t['elapsed']:.2f} 秒")


# ============================================================================
# 3. train 子命令
# ============================================================================

@cli.command(
    "train",
    help="训练分类模型（random_forest / fasttext / bert / all）",
)
@click.option(
    "--model",
    "model_type",
    type=click.Choice(["random_forest", "fasttext", "bert", "all"], case_sensitive=False),
    required=True,
    help="模型类型：random_forest(随机森林) / fasttext / bert / all(全部训练)",
)
@click.option(
    "--data",
    "data_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="预处理后的CSV数据路径（需含category_id和combined_text列）",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=lambda: MODEL_DIR,
    show_default="data/models",
    help="模型保存根目录",
)
@click.option(
    "--text-col",
    type=str,
    default="combined_text",
    show_default=True,
    help="用于训练的文本列名",
)
@click.option(
    "--label-col",
    type=str,
    default="category_id",
    show_default=True,
    help="标签列名（类目ID）",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="随机种子",
)
@click.option(
    "--bert-samples",
    type=int,
    default=0,
    show_default=True,
    help="BERT训练使用的样本数（0=全部；无GPU时建议设为小值如500）",
)
@click.option(
    "--bert-epochs",
    type=int,
    default=0,
    show_default=True,
    help="BERT训练轮数（0=使用配置文件值）",
)
def train(
    model_type: str,
    data_path: Path,
    output_dir: Path,
    text_col: str,
    label_col: str,
    seed: int,
    bert_samples: int,
    bert_epochs: int,
):
    """
    训练一个或多个分类模型。使用 'all' 可依次训练RF + FastText + BERT。

    \b
    示例：
        python run.py train --model random_forest --data data/processed/clean_data.csv
        python run.py train --model fasttext --data data/processed/clean.csv --output data/models/ft
        python run.py train --model all --data data/processed/clean_data.csv --bert-samples 500
    """
    seed_everything(seed)

    click.echo(f"\n{'='*60}")
    click.echo(f"🧠 开始模型训练")
    click.echo(f"   - 模型类型: {model_type.upper()}")
    click.echo(f"   - 训练数据: {data_path}")
    click.echo(f"   - 输出目录: {output_dir}")
    click.echo(f"   - 文本列: {text_col}")
    click.echo(f"   - 标签列: {label_col}")
    click.echo(f"{'='*60}\n")

    # 加载数据
    click.echo("📥 加载训练数据...")
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    if text_col not in df.columns:
        raise click.BadParameter(f"文本列 '{text_col}' 不存在于数据中")
    if label_col not in df.columns:
        raise click.BadParameter(f"标签列 '{label_col}' 不存在于数据中")

    X = df[text_col].fillna("").tolist()
    y = df[label_col].astype(int).tolist()
    click.echo(f"   样本数: {len(X)}, 类别数: {len(set(y))}")

    # 获取类目名称
    _, _, class_names = get_category_mappings()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_train = []
    if model_type.lower() == "all":
        models_to_train = ["random_forest", "fasttext", "bert"]
    else:
        models_to_train = [model_type.lower()]

    from src.models.trainer import Trainer

    total_start = time.time()

    for idx, m_type in enumerate(models_to_train, 1):
        click.echo(f"\n{'─'*60}")
        click.echo(f"🚀 训练模型 [{idx}/{len(models_to_train)}]: {m_type.upper()}")
        click.echo(f"{'─'*60}")

        try:
            with Timer(f"{m_type} 训练", verbose=False) as t:
                if m_type == "random_forest":
                    from src.models.random_forest import RandomForestClassifier
                    model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=30,
                        random_state=seed,
                    )
                    trainer = Trainer(output_dir=str(output_dir), class_names=class_names)
                    result = trainer.train(model, X, y, run_name="random_forest")

                elif m_type == "fasttext":
                    from src.models.fasttext_model import FastTextClassifier
                    model = FastTextClassifier(
                        epoch=30,
                        lr=0.1,
                        random_state=seed,
                    )
                    trainer = Trainer(output_dir=str(output_dir), class_names=class_names)
                    result = trainer.train(model, X, y, run_name="fasttext")

                elif m_type == "bert":
                    import torch
                    has_gpu = torch.cuda.is_available()

                    # 根据样本限制采样
                    X_bert = X
                    y_bert = y
                    if bert_samples > 0 and bert_samples < len(X):
                        click.echo(f"   ⚡ BERT训练采样 {bert_samples} 条（共{len(X)}条）")
                        from sklearn.model_selection import train_test_split
                        _, X_bert_s, _, y_bert_s = train_test_split(
                            X_bert, y_bert,
                            train_size=bert_samples / len(X_bert),
                            random_state=seed,
                            stratify=y_bert,
                        )
                        X_bert = X_bert_s
                        y_bert = y_bert_s

                    if not has_gpu and len(X_bert) > 2000:
                        click.echo(
                            f"   ⚠️  未检测到GPU，BERT训练将较慢。"
                            f"建议使用 --bert-samples 500 减少数据量。"
                        )

                    from src.models.bert_classifier import BERTClassifier
                    bert_kwargs = dict(
                        base_model="bert-base-chinese",
                        num_labels=12,
                        max_length=128,
                        batch_size=16 if has_gpu else 8,
                        num_train_epochs=bert_epochs if bert_epochs > 0 else (2 if not has_gpu else 3),
                        learning_rate=2.0e-5,
                        fp16=has_gpu,
                        device="cuda" if has_gpu else "cpu",
                        gradient_accumulation_steps=2,
                        random_state=seed,
                    )
                    model = BERTClassifier(model_name="bert", **bert_kwargs)
                    trainer = Trainer(output_dir=str(output_dir), class_names=class_names)
                    result = trainer.train(
                        model, X_bert, y_bert,
                        run_name="bert",
                    )

            # 输出结果
            test_metrics = result.get("test_metrics", {})
            f1 = test_metrics.get("f1_macro", 0)
            acc = test_metrics.get("accuracy", 0)
            click.echo(f"\n✅ {m_type.upper()} 训练完成！")
            click.echo(f"   - 测试集 F1-Macro: {f1:.4f}")
            click.echo(f"   - 测试集 Accuracy:  {acc:.4f}")
            click.echo(f"   - 训练耗时: {t['elapsed']:.1f} 秒")
            if "run_name" in result:
                click.echo(f"   - 模型保存: {output_dir / result['run_name']}")

        except Exception as e:
            logger.exception(f"训练 {m_type} 失败")
            click.echo(f"   ❌ 错误: {e}", err=True)

    total_elapsed = time.time() - total_start
    click.echo(f"\n{'='*60}")
    click.echo(f"🏁 全部训练流程完成 | 总耗时: {total_elapsed:.1f} 秒")
    click.echo(f"{'='*60}")


# ============================================================================
# 4. compress 子命令
# ============================================================================

@cli.command(
    "compress",
    help="模型压缩：量化/知识蒸馏/ONNX导出",
)
@click.option(
    "--model-path",
    "model_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="已训练的BERT模型目录路径",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="压缩后模型输出目录",
)
@click.option(
    "--method",
    type=click.Choice(["quantize", "distill", "onnx"], case_sensitive=False),
    required=True,
    help="压缩方法：quantize(量化) / distill(知识蒸馏) / onnx(导出ONNX)",
)
@click.option(
    "--data",
    "data_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="训练/校准数据CSV路径（蒸馏和静态量化需要）",
)
@click.option(
    "--text-col",
    type=str,
    default="combined_text",
    show_default=True,
    help="文本列名",
)
@click.option(
    "--label-col",
    type=str,
    default="category_id",
    show_default=True,
    help="标签列名",
)
@click.option(
    "--quant-type",
    type=click.Choice(["dynamic", "static", "qat"], case_sensitive=False),
    default="dynamic",
    show_default=True,
    help="量化类型（quantize方法用）：dynamic(动态)/static(静态)/qat(感知训练)",
)
@click.option(
    "--distill-epochs",
    type=int,
    default=3,
    show_default=True,
    help="蒸馏训练轮数（distill方法用）",
)
def compress(
    model_path: Path,
    output_dir: Path,
    method: str,
    data_path: Optional[Path],
    text_col: str,
    label_col: str,
    quant_type: str,
    distill_epochs: int,
):
    """
    对已训练的BERT模型进行压缩或导出。

    \b
    示例：
        python run.py compress --model-path data/models/bert --method onnx --output data/models/bert_onnx
        python run.py compress --model-path data/models/bert --method quantize --output data/models/bert_quant
        python run.py compress --model-path data/models/bert --method distill --data data/processed/clean.csv --output data/models/bert_distilled
    """
    click.echo(f"\n{'='*60}")
    click.echo(f"📦 开始模型压缩")
    click.echo(f"   - 方法: {method.upper()}")
    click.echo(f"   - 原始模型: {model_path}")
    click.echo(f"   - 输出目录: {output_dir}")
    click.echo(f"{'='*60}\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Timer(f"模型压缩({method})", verbose=False) as t:
        from src.models.bert_classifier import BERTClassifier
        from src.evaluation.model_compressor import (
            ModelCompressor,
            QuantizationConfig,
            QuantizationType,
            DistillationConfig,
        )

        # 加载原始BERT模型
        click.echo("📥 加载原始模型...")
        teacher = BERTClassifier(model_name="bert_teacher")
        teacher.load(str(model_path))
        click.echo(f"   模型加载完成 | 设备: {teacher.device}")

        _, _, class_names = get_category_mappings()
        compressor = ModelCompressor(
            class_names=class_names,
            default_output_dir=str(output_dir.parent),
        )

        X_calib = None
        y_calib = None
        if data_path and data_path.exists():
            click.echo(f"📊 加载校准/训练数据: {data_path}")
            df = pd.read_csv(data_path, encoding="utf-8-sig")
            X_calib = df[text_col].fillna("").tolist()
            y_calib = df[label_col].astype(int).tolist()
            click.echo(f"   数据量: {len(X_calib)} 条")

        if method.lower() == "quantize":
            qconfig = QuantizationConfig(
                quant_type=QuantizationType(quant_type.lower()),
            )
            click.echo(f"🔢 执行{quant_type.upper()}量化...")
            quantized, report = compressor.quantize(
                teacher,
                calibration_data=X_calib,
                output_dir=str(output_dir),
                config=qconfig,
                X_eval=X_calib,
                y_eval=y_calib,
            )
            click.echo(f"\n✅ 量化完成！")
            click.echo(f"   - 参数压缩率: {report.param_compression_ratio:.2f}x")
            click.echo(f"   - 体积压缩率: {report.size_compression_ratio:.2f}x")
            click.echo(f"   - 推理加速: {report.latency_speedup:.2f}x")
            click.echo(f"   - F1下降: {report.f1_macro_drop*100:.2f}个百分点")

        elif method.lower() == "distill":
            if X_calib is None:
                raise click.BadParameter("蒸馏方法需要提供 --data 训练数据")

            import torch
            has_gpu = torch.cuda.is_available()

            dconfig = DistillationConfig(
                temperature=4.0,
                alpha=0.5,
                num_train_epochs=distill_epochs,
                batch_size=16 if has_gpu else 8,
                device="cuda" if has_gpu else "cpu",
                fp16=has_gpu,
            )
            click.echo(f"🎓 执行知识蒸馏（{distill_epochs}轮）...")
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_calib, y_calib,
                test_size=0.1,
                random_state=42,
                stratify=y_calib,
            )
            student, report = compressor.distill(
                teacher,
                X_train, y_train,
                X_val, y_val,
                output_dir=str(output_dir),
                config=dconfig,
                X_eval=X_val,
                y_eval=y_val,
            )
            click.echo(f"\n✅ 蒸馏完成！")
            click.echo(f"   - 参数压缩率: {report.param_compression_ratio:.2f}x")
            click.echo(f"   - 体积压缩率: {report.size_compression_ratio:.2f}x")
            click.echo(f"   - 推理加速: {report.latency_speedup:.2f}x")
            click.echo(f"   - F1下降: {report.f1_macro_drop*100:.2f}个百分点")

        elif method.lower() == "onnx":
            click.echo("🔄 导出ONNX格式...")
            onnx_path = compressor.export_onnx(
                teacher,
                output_path=str(output_dir / "model.onnx"),
                dynamic_batch=True,
            )
            click.echo(f"\n✅ ONNX导出完成！")
            click.echo(f"   - ONNX文件: {onnx_path}")
            click.echo(f"   - 文件大小: {Path(onnx_path).stat().st_size/1024/1024:.2f} MB")

    click.echo(f"\n   - 总耗时: {t['elapsed']:.1f} 秒")


# ============================================================================
# 5. evaluate 子命令
# ============================================================================

@cli.command(
    "evaluate",
    help="评估模型，输出评估报告+混淆矩阵+可视化",
)
@click.option(
    "--model",
    "model_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="已训练模型的目录路径（支持random_forest/fasttext/bert）",
)
@click.option(
    "--data",
    "data_dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="测试数据目录（需含test.csv文件）或单个CSV文件路径",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=lambda: LOG_DIR / "evaluation",
    show_default="logs/evaluation",
    help="评估报告输出目录",
)
@click.option(
    "--text-col",
    type=str,
    default="combined_text",
    show_default=True,
    help="文本列名",
)
@click.option(
    "--label-col",
    type=str,
    default="category_id",
    show_default=True,
    help="标签列名",
)
@click.option(
    "--no-roc",
    is_flag=True,
    default=False,
    help="跳过ROC曲线绘制",
)
def evaluate(
    model_dir: Path,
    data_dir: Path,
    output_dir: Path,
    text_col: str,
    label_col: str,
    no_roc: bool,
):
    """
    评估已训练模型在测试集上的性能，输出指标报告、混淆矩阵图等。

    \b
    示例：
        python run.py evaluate --model data/models/random_forest/random_forest --data data/processed
        python run.py evaluate --model data/models/fasttext/fasttext --data data/processed/clean_data.csv --output logs/eval
    """
    click.echo(f"\n{'='*60}")
    click.echo(f"📈 开始模型评估")
    click.echo(f"   - 模型路径: {model_dir}")
    click.echo(f"   - 测试数据: {data_dir}")
    click.echo(f"   - 报告输出: {output_dir}")
    click.echo(f"{'='*60}\n")

    with Timer("模型评估", verbose=False) as t:
        from src.evaluation.evaluator import Evaluator

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 加载测试数据
        click.echo("📥 加载测试数据...")
        data_dir = Path(data_dir)
        if data_dir.is_dir():
            test_csv = data_dir / "test.csv"
            if not test_csv.exists():
                # 没有test.csv就用全部数据的后10%
                all_csv = list(data_dir.glob("*.csv"))
                if not all_csv:
                    raise click.BadParameter(f"数据目录 {data_dir} 中没有CSV文件")
                test_csv = all_csv[0]
        else:
            test_csv = data_dir

        click.echo(f"   使用测试文件: {test_csv}")
        df = pd.read_csv(test_csv, encoding="utf-8-sig")

        if text_col not in df.columns:
            raise click.BadParameter(f"文本列 '{text_col}' 不存在")
        if label_col not in df.columns:
            raise click.BadParameter(f"标签列 '{label_col}' 不存在")

        X_test = df[text_col].fillna("").tolist()
        y_test = df[label_col].astype(int).tolist()
        click.echo(f"   测试样本数: {len(X_test)}")

        # 加载模型（根据目录下的特征文件判断模型类型）
        click.echo("🧠 加载模型...")
        model = None
        model_name = Path(model_dir).name

        tfidf_path = Path(model_dir) / "tfidf.pkl"
        fasttext_bin = Path(model_dir) / "model.bin"
        bert_bin = Path(model_dir) / "pytorch_model.bin"

        if tfidf_path.exists():
            from src.models.random_forest import RandomForestClassifier
            model = RandomForestClassifier(model_name="random_forest")
            model.load(str(model_dir))
        elif fasttext_bin.exists():
            from src.models.fasttext_model import FastTextClassifier
            model = FastTextClassifier(model_name="fasttext")
            model.load(str(model_dir))
        elif bert_bin.exists():
            from src.models.bert_classifier import BERTClassifier
            model = BERTClassifier(model_name="bert")
            model.load(str(model_dir))
        else:
            raise click.BadParameter(
                f"无法识别模型类型（未找到 tfidf.pkl/model.bin/pytorch_model.bin）"
            )

        click.echo(f"   模型类型: {type(model).__name__}")

        # 评估
        _, _, class_names = get_category_mappings()
        evaluator = Evaluator(
            class_names=class_names,
            output_dir=str(output_dir),
            compute_roc=not no_roc,
            compute_pr=False,
        )

        click.echo("🔍 执行评估...")
        result = evaluator.evaluate(model, X_test, y_test, model_name=model_name)

        # 保存结果
        saved_paths = result.save(str(output_dir), prefix=model_name)

        # 保存可视化
        y_proba = None
        if not no_roc and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)

        vis_paths = evaluator.save_visualizations(
            result,
            output_dir=str(output_dir),
            save_cm=True,
            save_roc=not no_roc,
            y_true=y_test if not no_roc else None,
            y_prob=y_proba,
        )

    # 输出评估摘要
    click.echo(f"\n{'='*60}")
    click.echo(f"🏆 评估结果: {result.model_name}")
    click.echo(f"{'='*60}")
    click.echo(f"   Accuracy       : {result.accuracy:.4f}")
    click.echo(f"   Precision(Macro): {result.precision_macro:.4f}")
    click.echo(f"   Recall(Macro)  : {result.recall_macro:.4f}")
    click.echo(f"   F1(Macro) ★    : {result.f1_macro:.4f}")
    click.echo(f"   F1(Micro)      : {result.f1_micro:.4f}")
    click.echo(f"\n📁 已保存文件:")
    for k, v in saved_paths.items():
        click.echo(f"   - {k}: {v}")
    for k, v in vis_paths.items():
        click.echo(f"   - {k}: {v}")
    click.echo(f"\n⏱️  总耗时: {t['elapsed']:.1f} 秒")


# ============================================================================
# 6. serve 子命令
# ============================================================================

@cli.command(
    "serve",
    help="启动服务：API后端(FastAPI)或前端(Streamlit)",
)
@click.option(
    "--api",
    "serve_api",
    is_flag=True,
    default=False,
    help="启动FastAPI后端API服务",
)
@click.option(
    "--frontend",
    "serve_frontend",
    is_flag=True,
    default=False,
    help="启动Streamlit前端运营后台",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="服务端口（API默认8000，前端默认8501）",
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    show_default=True,
    help="服务绑定主机地址",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="启用热重载（开发模式用）",
)
def serve(serve_api: bool, serve_frontend: bool, port: int, host: str, reload: bool):
    """
    启动后端API服务或前端运营后台。

    \b
    示例：
        python run.py serve --api --port 8000
        python run.py serve --frontend --port 8501
        python run.py serve --api --frontend      # 同时启动（后台方式，需自行管理）
    """
    if not serve_api and not serve_frontend:
        raise click.UsageError("请至少指定 --api 或 --frontend")

    default_api_port = 8000
    default_frontend_port = 8501

    if serve_api and serve_frontend:
        if port is not None:
            click.echo("⚠️ 同时启动API和前端时--port将被忽略，使用默认端口8000和8501")
        api_port = default_api_port
        fe_port = default_frontend_port
    else:
        if serve_api:
            api_port = port or default_api_port
        if serve_frontend:
            fe_port = port or default_frontend_port

    # 启动API服务
    if serve_api:
        click.echo(f"\n🚀 启动 FastAPI 后端服务 | {host}:{api_port}")
        click.echo(f"   API文档: http://localhost:{api_port}/docs")
        click.echo(f"   健康检查: http://localhost:{api_port}/health")
        click.echo(f"   按 Ctrl+C 停止服务\n")

        import uvicorn
        uvicorn.run(
            "app.api.main:app",
            host=host,
            port=api_port,
            reload=reload,
            log_level="info",
        )

    # 启动前端（此分支仅在单独启动前端时到达，同时启动会被uvicorn阻塞）
    if serve_frontend and not serve_api:
        click.echo(f"\n🎨 启动 Streamlit 前端 | {host}:{fe_port}")
        click.echo(f"   访问地址: http://localhost:{fe_port}")
        click.echo(f"   按 Ctrl+C 停止服务\n")

        import subprocess
        app_path = PROJECT_ROOT / "app" / "frontend" / "app.py"
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            f"--server.port={fe_port}",
            f"--server.address={host}",
            f"--server.headless=true",
        ]
        if reload:
            cmd.append("--server.runOnSave=true")
        subprocess.run(cmd)


# ============================================================================
# 7. pipeline 子命令
# ============================================================================

@cli.command(
    "pipeline",
    help="一键跑通全流程：生成数据→预处理→训练RF+FastText→评估→启动服务",
)
@click.option(
    "--samples",
    "samples_per_category",
    type=int,
    default=100,
    show_default=True,
    help="每个类目样本数（默认100，快速跑通）",
)
@click.option(
    "--no-eda",
    is_flag=True,
    default=False,
    help="跳过EDA分析",
)
@click.option(
    "--no-bert",
    is_flag=True,
    default=False,
    help="跳过BERT训练（推荐无GPU时开启，默认已跳过BERT）",
)
@click.option(
    "--with-bert",
    is_flag=True,
    default=False,
    help="启用BERT训练（无GPU将使用CPU且限制样本）",
)
@click.option(
    "--bert-samples",
    type=int,
    default=500,
    show_default=True,
    help="BERT训练样本数（无GPU时限制总量）",
)
@click.option(
    "--serve",
    "start_serve",
    is_flag=True,
    default=False,
    help="流程完成后自动启动API+前端服务",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="随机种子",
)
def pipeline(
    samples_per_category: int,
    no_eda: bool,
    no_bert: bool,
    with_bert: bool,
    bert_samples: int,
    start_serve: bool,
    seed: int,
):
    """
    一键端到端流水线：数据生成 → 预处理 → 训练RF+FastText → 评估 → 可选启动服务。

    \b
    示例：
        python run.py pipeline                                  # 快速端到端，每类100条，无BERT
        python run.py pipeline --samples 200 --with-bert        # 含BERT训练
        python run.py pipeline --samples 500 --serve            # 完成后启动服务
    """
    pipeline_start = time.time()

    # 路径配置
    raw_data = RAW_DATA_DIR / "pipeline_mock.csv"
    clean_data = PROCESSED_DATA_DIR / "pipeline_clean.csv"
    models_root = MODEL_DIR
    eval_output = LOG_DIR / "pipeline_eval"

    # 决定是否训练BERT
    do_bert = with_bert and not no_bert
    import torch
    has_gpu = torch.cuda.is_available()

    steps = []
    steps.append(("1/6", "数据生成", f"每类{samples_per_category}条"))
    steps.append(("2/6", "数据预处理", f"{'含EDA' if not no_eda else '不含EDA'}"))
    steps.append(("3/6", "训练RandomForest", "200棵树"))
    steps.append(("4/6", "训练FastText", "30轮"))
    if do_bert:
        steps.append(("5/6", "训练BERT", f"{bert_samples}样本 | {'GPU' if has_gpu else 'CPU'}"))
    steps.append(("6/6", "评估+报告", "混淆矩阵+指标"))

    click.echo(f"\n{'═'*70}")
    click.echo(f" 🏭 商品分类系统 - 一键端到端训练流水线")
    click.echo(f"{'═'*70}")
    click.echo(f"\n📋 执行计划：")
    for step_id, step_name, step_detail in steps:
        click.echo(f"   {step_id} {step_name:<20s} | {step_detail}")
    click.echo(f"\n{'─'*70}\n")

    results = {}

    # ------------------------------------------------------------
    # Step 1: 数据生成
    # ------------------------------------------------------------
    click.echo("\n" + "="*60)
    click.echo("📊 Step 1/6: 生成模拟数据")
    click.echo("="*60)
    with Timer("数据生成", verbose=False) as t:
        from src.data.data_loader import MockDataGenerator, load_categories_config

        categories = load_categories_config()
        gen = MockDataGenerator(
            categories=categories,
            random_seed=seed,
            samples_per_category=samples_per_category,
        )
        df_raw = gen.generate()
        raw_data.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(raw_data, index=False, encoding="utf-8-sig")

    click.echo(f"✅ 数据生成完成: {len(df_raw)}条 | {t['elapsed']:.1f}s")
    results["step1"] = {"name": "数据生成", "count": len(df_raw), "time": t["elapsed"]}

    # ------------------------------------------------------------
    # Step 2: 预处理
    # ------------------------------------------------------------
    click.echo("\n" + "="*60)
    click.echo("🔧 Step 2/6: 数据预处理")
    click.echo("="*60)
    with Timer("预处理", verbose=False) as t:
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        df_clean = preprocessor.transform(df_raw)
        clean_data.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(clean_data, index=False, encoding="utf-8-sig")

        if not no_eda:
            try:
                from src.data.eda import run_full_eda
                eda_dir = LOG_DIR / "pipeline_eda"
                eda_dir.mkdir(parents=True, exist_ok=True)
                run_full_eda(df_clean, output_dir=str(eda_dir))
                click.echo(f"   EDA报告: {eda_dir}")
            except Exception as e:
                logger.warning(f"EDA失败（可忽略）: {e}")

    click.echo(f"✅ 预处理完成: {len(df_clean)}条 | {t['elapsed']:.1f}s")
    results["step2"] = {"name": "预处理", "count": len(df_clean), "time": t["elapsed"]}

    # ------------------------------------------------------------
    # 准备训练数据
    # ------------------------------------------------------------
    X = df_clean["combined_text"].fillna("").tolist()
    y = df_clean["category_id"].astype(int).tolist()
    _, _, class_names = get_category_mappings()
    from src.models.trainer import Trainer

    # ------------------------------------------------------------
    # Step 3: RandomForest
    # ------------------------------------------------------------
    click.echo("\n" + "="*60)
    click.echo("🌲 Step 3/6: 训练 RandomForest")
    click.echo("="*60)
    with Timer("RF训练", verbose=False) as t:
        from src.models.random_forest import RandomForestClassifier
        rf_model = RandomForestClassifier(
            model_name="random_forest",
            n_estimators=200,
            max_depth=30,
            random_state=seed,
        )
        trainer_rf = Trainer(output_dir=str(models_root), class_names=class_names)
        rf_result = trainer_rf.train(rf_model, X, y, run_name="random_forest_pipeline")

    rf_f1 = rf_result.get("test_metrics", {}).get("f1_macro", 0)
    click.echo(f"✅ RF训练完成 | F1={rf_f1:.4f} | {t['elapsed']:.1f}s")
    results["step3"] = {"name": "RF训练", "f1": rf_f1, "time": t["elapsed"]}

    # ------------------------------------------------------------
    # Step 4: FastText
    # ------------------------------------------------------------
    click.echo("\n" + "="*60)
    click.echo("⚡ Step 4/6: 训练 FastText")
    click.echo("="*60)
    with Timer("FT训练", verbose=False) as t:
        from src.models.fasttext_model import FastTextClassifier
        ft_model = FastTextClassifier(
            model_name="fasttext",
            epoch=30,
            lr=0.1,
            random_state=seed,
        )
        trainer_ft = Trainer(output_dir=str(models_root), class_names=class_names)
        ft_result = trainer_ft.train(ft_model, X, y, run_name="fasttext_pipeline")

    ft_f1 = ft_result.get("test_metrics", {}).get("f1_macro", 0)
    click.echo(f"✅ FT训练完成 | F1={ft_f1:.4f} | {t['elapsed']:.1f}s")
    results["step4"] = {"name": "FT训练", "f1": ft_f1, "time": t["elapsed"]}

    # ------------------------------------------------------------
    # Step 5: BERT（可选）
    # ------------------------------------------------------------
    if do_bert:
        step5_id = "5/6"
        step6_id = "6/6"
        click.echo("\n" + "="*60)
        click.echo("🧠 Step 5/6: 训练 BERT")
        if not has_gpu:
            click.echo("   (无GPU，将使用CPU且限制样本数)")
        click.echo("="*60)

        with Timer("BERT训练", verbose=False) as t:
            from sklearn.model_selection import train_test_split
            actual_bert_samples = min(bert_samples, len(X))
            if actual_bert_samples < len(X):
                _, X_b, _, y_b = train_test_split(
                    X, y,
                    train_size=actual_bert_samples/len(X),
                    random_state=seed,
                    stratify=y,
                )
            else:
                X_b, y_b = X, y

            from src.models.bert_classifier import BERTClassifier
            bert_model = BERTClassifier(
                model_name="bert",
                base_model="bert-base-chinese",
                num_labels=12,
                max_length=128,
                batch_size=8 if not has_gpu else 16,
                num_train_epochs=2 if not has_gpu else 3,
                learning_rate=2e-5,
                fp16=has_gpu,
                device="cuda" if has_gpu else "cpu",
                random_state=seed,
            )
            trainer_bert = Trainer(output_dir=str(models_root), class_names=class_names)
            bert_result = trainer_bert.train(
                bert_model, X_b, y_b, run_name="bert_pipeline",
            )

        bert_f1 = bert_result.get("test_metrics", {}).get("f1_macro", 0)
        click.echo(f"✅ BERT训练完成 | F1={bert_f1:.4f} | {t['elapsed']:.1f}s")
        results["step5"] = {"name": "BERT训练", "f1": bert_f1, "time": t["elapsed"]}
    else:
        step5_id = "5/6(跳过)"
        step6_id = "6/6"
        click.echo("\n" + "="*60)
        click.echo("🧠 Step 5/6: BERT 训练 → 跳过 (使用 --with-bert 启用)")
        click.echo("="*60)
        results["step5"] = {"name": "BERT训练(跳过)", "f1": None, "time": 0}

    # ------------------------------------------------------------
    # Step 6: 评估
    # ------------------------------------------------------------
    click.echo("\n" + "="*60)
    click.echo("📈 Step 6/6: 模型评估 + 报告生成")
    click.echo("="*60)
    with Timer("评估", verbose=False) as t:
        from src.evaluation.evaluator import Evaluator

        eval_output.mkdir(parents=True, exist_ok=True)
        evaluator = Evaluator(
            class_names=class_names,
            output_dir=str(eval_output),
            compute_roc=True,
        )

        eval_results = []

        # RF评估
        click.echo("   评估 RandomForest...")
        rf_eval = evaluator.evaluate(rf_model, X, y, model_name="RandomForest")
        eval_results.append(rf_eval)
        rf_eval.save(str(eval_output), prefix="rf")

        # FT评估
        click.echo("   评估 FastText...")
        ft_eval = evaluator.evaluate(ft_model, X, y, model_name="FastText")
        eval_results.append(ft_eval)
        ft_eval.save(str(eval_output), prefix="ft")

        # 对比汇总
        compare_df = evaluator.compare_models(
            eval_results,
            output_path=str(eval_output / "model_comparison.csv"),
        )

        # 混淆矩阵可视化
        for ev in eval_results:
            try:
                evaluator.save_visualizations(
                    ev, output_dir=str(eval_output),
                    save_cm=True, save_roc=False,
                )
            except Exception:
                pass

    click.echo(f"✅ 评估完成 | {t['elapsed']:.1f}s")
    results["step6"] = {"name": "评估", "time": t["elapsed"]}

    # ------------------------------------------------------------
    # 汇总报告
    # ------------------------------------------------------------
    total_time = time.time() - pipeline_start
    click.echo(f"\n{'═'*70}")
    click.echo(f" 🏁 端到端流水线完成 | 总耗时: {total_time:.1f}秒")
    click.echo(f"{'═'*70}")
    click.echo(f"\n📊 各步骤耗时 & 性能：")
    for key, info in results.items():
        if "f1" in info and info["f1"] is not None:
            line = f"   {info['name']:<25s} | {info['time']:>6.1f}s | F1={info['f1']:.4f}"
        else:
            line = f"   {info['name']:<25s} | {info['time']:>6.1f}s"
        click.echo(line)

    click.echo(f"\n📁 产物位置：")
    click.echo(f"   - 原始数据: {raw_data}")
    click.echo(f"   - 清洗数据: {clean_data}")
    click.echo(f"   - RF模型:   {models_root / 'random_forest_pipeline'}")
    click.echo(f"   - FT模型:   {models_root / 'fasttext_pipeline'}")
    click.echo(f"   - 评估报告: {eval_output}")

    if start_serve:
        click.echo(f"\n🚀 5秒后启动服务（API:8000 + 前端:8501）...")
        time.sleep(5)

        import subprocess
        import signal

        api_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn",
             "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=str(PROJECT_ROOT),
        )
        fe_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run",
             str(PROJECT_ROOT / "app/frontend/app.py"),
             "--server.port=8501", "--server.headless=true"],
            cwd=str(PROJECT_ROOT),
        )

        click.echo(f"\n✅ 服务已启动！")
        click.echo(f"   API后端:  http://localhost:8000/docs")
        click.echo(f"   前端后台: http://localhost:8501")
        click.echo(f"   按 Ctrl+C 停止所有服务")

        try:
            api_proc.wait()
            fe_proc.wait()
        except KeyboardInterrupt:
            api_proc.terminate()
            fe_proc.terminate()
            click.echo("\n服务已停止")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    cli()
